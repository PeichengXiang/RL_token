"""Training and evaluation entry points for the reproduction."""

import argparse
from copy import copy
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from .online import Learner, OnlineConfig
from .rollout import OnlineTrainer, RolloutConfig, run_episode
from .stage1 import mean_reconstruction_mse, read_features, train_token, write_features
from .token import RLToken, TokenConfig, load_token, save_token


def emit(data):
    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def fresh_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Choose an empty output directory: {path}")


def write_checkpoint(path, state):
    pending = path.with_suffix(".pt.tmp")
    torch.save(state, pending)
    pending.replace(path)


def evaluation(env, vla, token, rollout, seeds, learner=None, **kwargs):
    policy = None if learner is None else partial(learner.act, deterministic=True)
    episodes = [
        {"seed": int(seed), **asdict(run_episode(env, vla, token, rollout, seed, policy=policy, **kwargs))}
        for seed in seeds
    ]
    successful_steps = [trial["steps"] for trial in episodes if trial["success"]]
    successes = len(successful_steps)
    steps = sum(trial["steps"] for trial in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "mean_steps": steps / len(episodes),
        "mean_steps_success_only": sum(successful_steps) / successes if successes else None,
        "successes_per_10min_sim_time": successes * env.control_hz * 600 / steps,
        "trial_results": episodes,
    }


def run_training(env, vla, token, rollout, learner, args, *, save_checkpoint=None,
                 expert_replay=None, resume=None, save_resume=None, **kwargs):
    trainer = OnlineTrainer(
        learner, seed=args.seed, warmup_steps=args.warmup_steps,
        updates_per_transition=args.utd, asynchronous=not args.sync,
    )
    completed = 0
    counts = {"successes": 0, "actor_episodes": 0, "actor_successes": 0}
    expert_summary = {}
    try:
        if resume is not None:
            completed = resume["episodes_completed"]
            counts = resume["counts"].copy()
            expert_summary = resume["expert_summary"].copy()
            learner.load_state_dict(resume.pop("learner"))
            trainer.load_state_dict(resume.pop("trainer"))
            torch.set_rng_state(resume["torch_rng"])
            if learner.device.type == "cuda":
                torch.cuda.set_rng_state(resume["cuda_rng"], learner.device)
            emit({"resumed_after_episodes": completed, **counts,
                  "critic_updates": learner.critic_updates, "actor_updates": learner.actor_updates})
        elif expert_replay is not None:
            demo_seeds = []
            positive_windows = 0
            for seed, transitions in expert_replay:
                demo_seeds.append(seed)
                positive_windows += sum(transition.reward > 0 for transition in transitions)
                for transition in transitions:
                    trainer.replay.add(transition)
                emit({
                    "expert_demo_seed": seed, "expert_windows_added": len(transitions),
                    "expert_positive_windows_so_far": positive_windows,
                })
            if positive_windows == 0:
                raise ValueError("Expert replay must contain verified positive-reward transitions")
            sampled_positive_windows = 0
            for _ in range(args.expert_bootstrap_updates):
                batch = trainer.replay.sample(learner.config.batch_size)
                sampled_positive_windows += int(np.count_nonzero(batch.reward > 0))
                trainer.metrics.update(learner.update(batch))
            expert_summary = {
                "expert_demo_seeds": demo_seeds,
                "expert_condition_vla": getattr(args, "expert_condition_vla", False),
                "expert_replay_windows": len(trainer.replay),
                "expert_positive_windows": positive_windows,
                "expert_positive_windows_sampled": sampled_positive_windows,
                "expert_bootstrap_critic_updates": args.expert_bootstrap_updates,
            }
            emit({
                **expert_summary, "critic_updates": learner.critic_updates,
                "actor_updates": learner.actor_updates, **trainer.metrics,
            })
            if save_checkpoint is not None:
                save_checkpoint(0)
        if save_resume is not None and resume is None:
            save_resume(0, counts, expert_summary, trainer)
        for index in range(completed, args.episodes):
            episode = trainer.train_episode(env, vla, token, rollout, args.seed + index, **kwargs)
            counts["successes"] += episode.success
            if episode.actor_steps:
                counts["actor_episodes"] += 1
                counts["actor_successes"] += episode.success
            count = index + 1
            checkpoint_due = save_checkpoint is not None and count % args.checkpoint_every == 0
            resume_due = save_resume is not None and (
                count == 1 or count % args.resume_every == 0 or count == args.episodes
            )
            if index == 0 or count % 10 == 0 or count == args.episodes or checkpoint_due or resume_due:
                trainer.finish()
                emit({
                    "episode": count, **asdict(episode),
                    "critic_updates": learner.critic_updates, "actor_updates": learner.actor_updates,
                    **trainer.metrics,
                })
            if checkpoint_due and count < args.episodes:
                save_checkpoint(count)
            if resume_due:
                save_resume(count, counts, expert_summary, trainer)
    finally:
        trainer.close()
    return {
        "episodes": args.episodes, "critical_steps": trainer.critical_steps,
        "transitions": trainer.transitions, "critic_updates": learner.critic_updates,
        "actor_updates": learner.actor_updates, **counts, **expert_summary,
    }


def demo(args):
    from .toy import SlotEnv, ToyVLA, toy_demonstrations

    fresh_directory(args.out)
    torch.manual_seed(args.seed)
    vla, env = ToyVLA(), SlotEnv()
    train = toy_demonstrations(vla, 512, args.seed)
    held_out = toy_demonstrations(vla, 64, args.seed + 901)
    token_path = args.out / "token.pt"
    model = RLToken(TokenConfig(
        vla.feature_dim, token_dim=32, max_tokens=vla.feature_count,
        encoder_layers=1, decoder_layers=1, heads=4, ff_multiplier=2,
    ))
    before = mean_reconstruction_mse(model, held_out)
    train_token(model, train.__getitem__, len(train), steps=args.token_steps, batch_size=32,
                lr=1e-3, seed=args.seed, report=emit)
    save_token(token_path, model)
    token = load_token(token_path)
    stage1 = {"heldout_mse_before": before, "heldout_mse_after": mean_reconstruction_mse(token, held_out)}
    rollout = RolloutConfig()
    settings = OnlineConfig(hidden_dim=128, batch_size=64, beta=0.3, action_std=0.03)
    learner = Learner(
        token.config.token_dim + env.proprio_dim, rollout.chunk_size, env.action_dim,
        settings, action_projection=env.action_projection,
    )
    seeds = list(range(args.seed + 10_000, args.seed + 10_000 + args.eval_episodes))
    base = evaluation(env, vla, None, rollout, seeds)
    trained = run_training(env, vla, token, rollout, learner, args)
    torch.save({"learner": learner.state_dict(), "rollout": asdict(rollout)}, args.out / "learner.pt")
    final = evaluation(env, vla, token, rollout, seeds, learner)
    summary = {"seed": args.seed, "stage1": stage1, "training": trained, "base": base, "rlt": final}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    emit(summary)


def tianji_vla(args):
    from .openpi import FrozenOpenPI

    return FrozenOpenPI.from_tianji(
        args.vla_checkpoint, include_language=args.include_language,
        num_steps=args.flow_steps, seed=args.seed, shared_prefix=getattr(args, "shared_prefix", False),
    )


def cache_tianji(args):
    from .tianji import iter_demonstrations

    vla = tianji_vla(args)
    observations = iter_demonstrations(
        args.dataset, args.level, stride=args.frame_stride, max_episodes=args.max_episodes,
        seed=args.seed,
    )
    count = write_features(args.out, vla, observations, limit=args.max_frames)
    provenance = {
        "vla_checkpoint": str(args.vla_checkpoint.resolve()),
        "include_language": args.include_language, "dataset": str(args.dataset.resolve()),
        "level": args.level, "frame_stride": args.frame_stride, "seed": args.seed, "frames": count,
    }
    (args.out / "source.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    emit(provenance)


def fit_token(args):
    files = sorted(args.cache.glob("*.npz"))
    if not files:
        raise ValueError(f"No cached feature frames in {args.cache}")
    fresh_directory(args.out)
    torch.manual_seed(args.seed)
    samples = partial(_read_sample, files)
    input_features, _ = samples(0)
    config = TokenConfig(
        input_features.shape[1], token_dim=args.token_dim, max_tokens=args.max_tokens,
        encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, heads=args.heads,
        ff_multiplier=args.ff_multiplier,
    )
    model = RLToken(config).to(args.device)
    train_token(
        model, samples, len(files), steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, report=emit,
    )
    save_token(args.out / "token.pt", model)
    provenance = args.cache / "source.json"
    if provenance.is_file():
        (args.out / "source.json").write_text(provenance.read_text(encoding="utf-8"), encoding="utf-8")


def _read_sample(files, index):
    return read_features(files[index])


def _validate_tianji_token(args):
    metadata = args.token.parent / "source.json"
    if not metadata.is_file():
        return
    source = json.loads(metadata.read_text(encoding="utf-8"))
    if source.get("level", args.level) != args.level:
        raise ValueError(f"Token was trained for {source['level']!r}; requested task is {args.level!r}")
    if source["include_language"] != args.include_language:
        raise ValueError("The online language mask must match the token training cache")
    if "run_id" in source:
        config = args.vla_checkpoint / "training_config.json"
        vla = json.loads(config.read_text(encoding="utf-8"))
        if source["run_id"] != vla.get("run_id"):
            raise ValueError(f"Token uses VLA run {source['run_id']!r}; checkpoint belongs to {vla.get('run_id')!r}")


def _initial_tianji_learner(args, rollout, settings):
    checkpoint = torch.load(args.init_learner, map_location="cpu", weights_only=True)
    expected = {
        "level": args.level,
        "vla_checkpoint": str(args.vla_checkpoint.resolve()),
        "token": str(args.token.resolve()),
        "critical_start_step": args.critical_start_step or 0,
        "critical_phase": args.critical_phase,
        "vla_seed": args.vla_seed,
    }
    for key in ("warmup_steps", "max_steps"):
        if key in checkpoint:
            expected[key] = getattr(args, key)
    for key, value in expected.items():
        if key not in checkpoint or checkpoint[key] != value:
            raise ValueError(f"Initial learner {key} differs from the requested training configuration")
    if RolloutConfig(**checkpoint["rollout"]) != rollout:
        raise ValueError("Initial learner rollout differs from the requested training configuration")
    state = checkpoint["learner"]
    if state["config"] != asdict(settings):
        raise ValueError("Initial learner settings differ from the requested training settings")
    return state


def tianji_run(args):
    from .tianji import TianjiEnv, ready_phase

    training = args.command == "train-tianji"
    resume = None
    if training:
        training_path = args.out / "training.pt"
        plan = {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"run", "resume", "resume_every", "checkpoint_every", "init_learner"}
        }
        if args.resume:
            resume = torch.load(training_path, map_location="cpu", weights_only=True, mmap=True)
            if resume["plan"] != plan:
                raise ValueError("Training checkpoint configuration differs from the requested run")
            if not 0 <= resume["episodes_completed"] <= args.episodes:
                raise ValueError("Training checkpoint is outside the requested episode budget")
        else:
            fresh_directory(args.out)
        emit({"level": args.level, "target_episodes": args.episodes, "seed": args.seed,
              "warmup_steps": args.warmup_steps, "critical_phase": args.critical_phase})
    torch.manual_seed(args.seed)
    token = None
    if args.token:
        _validate_tianji_token(args)
        token = load_token(args.token, args.device)

    checkpoint = None
    rollout = RolloutConfig(
        args.chunk_size, args.stride, args.base_chunk_size, args.gamma,
        warmup_chunk_size=getattr(args, "warmup_chunk_size", None),
    )
    phase_start = args.critical_start_step
    phase_name = args.critical_phase
    vla_seed = args.vla_seed
    if not training and args.learner:
        if token is None:
            raise ValueError("--learner requires --token")
        checkpoint = torch.load(args.learner, map_location="cpu", weights_only=True)
        if checkpoint.get("level", args.level) != args.level:
            raise ValueError(f"Learner was trained for {checkpoint['level']!r}; requested task is {args.level!r}")
        rollout = RolloutConfig(**checkpoint["rollout"])
        if phase_start is None and phase_name is None:
            phase_start = checkpoint["critical_start_step"]
            phase_name = checkpoint.get("critical_phase")
        if vla_seed is None:
            vla_seed = checkpoint.get("vla_seed")
    phase_start = phase_start or 0
    phase = None
    if phase_name == "ready":
        phase = ready_phase
    elif phase_start:
        def phase(observation, info):
            return info.get("steps", 0) >= phase_start

    initial_state = None
    if training:
        settings = OnlineConfig(
            hidden_dim=512 if args.network == "screw" else 256,
            hidden_layers=3 if args.network == "screw" else 2,
            actor_lr=args.actor_lr, critic_lr=args.critic_lr, beta=args.beta,
            action_std=args.action_std, tau=args.tau, batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
        )
        if args.init_learner is not None:
            initial_state = _initial_tianji_learner(args, rollout, settings)

    vla = tianji_vla(args)
    with TianjiEnv(args.level, max_steps=args.max_steps, benchmark_root=args.benchmark) as env:
        shared = {"phase": phase, "action_offset": env.action_offset, "vla_seed": vla_seed}
        if training:
            learner = Learner(
                token.config.token_dim + env.proprio_dim, rollout.chunk_size, env.action_dim,
                settings, device=args.device, action_projection=env.action_projection(args.device),
            )
            if initial_state is not None:
                learner.load_state_dict(initial_state)
                initialization = {
                    "init_learner": str(args.init_learner.resolve()),
                    "initial_critic_updates": learner.critic_updates,
                    "initial_actor_updates": learner.actor_updates,
                    "warmup_steps": args.warmup_steps,
                }
                del initial_state
                emit(initialization)
            else:
                initialization = resume["initialization"] if resume is not None else {}

            def save_checkpoint(episodes_completed):
                write_checkpoint(args.out / "learner.pt", {
                    "learner": learner.state_dict(), "rollout": asdict(rollout),
                    "critical_start_step": phase_start, "level": args.level,
                    "critical_phase": phase_name, "vla_seed": vla_seed,
                    "vla_checkpoint": str(args.vla_checkpoint.resolve()),
                    "token": str(args.token.resolve()), "training_episodes": episodes_completed,
                    "warmup_steps": args.warmup_steps, "max_steps": args.max_steps,
                    **initialization,
                })

            def save_resume(episodes_completed, counts, expert_summary, trainer):
                write_checkpoint(training_path, {
                    "plan": plan, "episodes_completed": episodes_completed,
                    "counts": counts, "expert_summary": expert_summary,
                    "initialization": initialization,
                    "learner": learner.state_dict(), "trainer": trainer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(learner.device) if learner.device.type == "cuda" else None,
                })
                emit({"replay_checkpoint_episode": episodes_completed})

            expert_replay = None
            if args.expert_demos and resume is None:
                from .demo_replay import iter_expert_replay

                expert_replay = iter_expert_replay(
                    args.expert_dataset or args.benchmark / "benchmark_data",
                    args.level, copy(vla), token, rollout,
                    max_episodes=args.expert_demos, stride=args.expert_stride,
                    seed=args.seed, phase=phase_name,
                    benchmark_root=args.benchmark,
                    condition_on_vla=args.expert_condition_vla,
                )
            replay_vla = copy(vla) if args.separate_replay_rng or args.warmup_chunk_size else None
            summary = run_training(
                env, vla, token, rollout, learner, args,
                save_checkpoint=save_checkpoint, replay_vla=replay_vla, expert_replay=expert_replay,
                resume=resume, save_resume=save_resume if args.resume_every else None,
                **shared,
            )
            save_checkpoint(args.episodes)
            summary.update(initialization)
            pending_summary = args.out / "summary.json.tmp"
            pending_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            pending_summary.replace(args.out / "summary.json")
            emit(summary)
        else:
            seeds = list(range(args.seed, args.seed + args.episodes))
            learner = None
            if checkpoint is not None:
                state = checkpoint["learner"]
                learner = Learner(
                    token.config.token_dim + env.proprio_dim, rollout.chunk_size, env.action_dim,
                    OnlineConfig(**state["config"]), device=args.device,
                    action_projection=env.action_projection(args.device),
                )
                learner.load_state_dict(state)
            metrics = evaluation(env, vla, token, rollout, seeds, learner, **shared)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.open("x", encoding="utf-8") as output:
                    output.write(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
            emit(metrics)



def training_args(parser, default_episodes, warmup):
    parser.add_argument("--episodes", type=int, default=default_episodes)
    parser.add_argument("--warmup-steps", type=int, default=warmup)
    parser.add_argument("--utd", type=int, default=5)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--out", type=Path, required=True)


def backbone_args(parser):
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--include-language", action="store_true")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--level", default="timed_single")
    parser.add_argument("--seed", type=int, default=0)


def main():
    from .tianji import DEFAULT_BENCHMARK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-threads", type=int, default=1)
    commands = parser.add_subparsers(dest="command", required=True)
    toy = commands.add_parser("demo", help="Validate both stages on a synthetic slot task")
    training_args(toy, 100, 200)
    toy.add_argument("--token-steps", type=int, default=300)
    toy.add_argument("--eval-episodes", type=int, default=50)
    toy.add_argument("--seed", type=int, default=7)
    toy.set_defaults(run=demo)

    cache = commands.add_parser("cache-tianji", help="Extract frozen VLA embeddings from HDF5 demonstrations")
    backbone_args(cache)
    cache.add_argument("--dataset", type=Path, default=DEFAULT_BENCHMARK / "benchmark_data")
    cache.add_argument("--out", type=Path, required=True)
    cache.add_argument("--frame-stride", type=int, default=10)
    cache.add_argument("--max-episodes", type=int)
    cache.add_argument("--max-frames", type=int)
    cache.set_defaults(run=cache_tianji)

    token = commands.add_parser("train-token", help="Fit the Stage 1 bottleneck from cached features")
    token.add_argument("--cache", type=Path, required=True)
    token.add_argument("--out", type=Path, required=True)
    token.add_argument("--steps", type=int, default=2000)
    token.add_argument("--batch-size", type=int, default=2)
    token.add_argument("--token-dim", type=int, default=2048)
    token.add_argument("--max-tokens", type=int, default=1024)
    token.add_argument("--heads", type=int, default=8)
    token.add_argument("--encoder-layers", type=int, default=2)
    token.add_argument("--decoder-layers", type=int, default=2)
    token.add_argument("--ff-multiplier", type=int, default=4)
    token.add_argument("--lr", type=float, default=3e-4)
    token.add_argument("--device", default="cpu")
    token.add_argument("--seed", type=int, default=0)
    token.set_defaults(run=fit_token)

    online = commands.add_parser("train-tianji", help="Run Stage 2 inside the local Isaac insertion benchmark")
    training_args(online, 400, 1000)
    online.add_argument("--network", choices=("default", "screw"), default="default")
    online.add_argument("--action-std", type=float, default=0.05)
    online.add_argument("--actor-lr", type=float, default=3e-4)
    online.add_argument("--critic-lr", type=float, default=3e-4)
    online.add_argument("--beta", type=float, default=0.1)
    online.add_argument("--tau", type=float, default=0.005)
    online.add_argument("--batch-size", type=int, default=256)
    online.add_argument("--replay-capacity", type=int, default=100000)
    online.add_argument("--checkpoint-every", type=int, default=10,
                        help="Update learner.pt atomically after every N completed episodes")
    initialization = online.add_mutually_exclusive_group()
    initialization.add_argument("--init-learner", type=Path,
                                help="Load actor, critics, optimizers and counters; start new replay and warmup")
    initialization.add_argument("--resume", action="store_true",
                                help="Resume the same run from its durable training.pt checkpoint")
    online.add_argument("--resume-every", type=int, default=0,
                        help="Atomically save replay and training state every N episodes; 0 disables it")
    online.add_argument("--separate-replay-rng", action="store_true",
                        help="Keep intermediate replay queries from consuming behavior samples")
    online.add_argument("--warmup-chunk-size", type=int,
                        help="Execute this many frozen VLA actions at a time during warmup")
    online.add_argument("--expert-demos", type=int, default=0,
                        help="Prefill replay with verified training-split expert successes")
    online.add_argument("--expert-dataset", type=Path,
                        help="Native benchmark demo directory; defaults to BENCHMARK/benchmark_data")
    online.add_argument("--expert-stride", type=int, default=20,
                        help="Sample nonterminal expert windows this many control ticks apart")
    online.add_argument("--expert-bootstrap-updates", type=int, default=200,
                        help="Critic updates on expert replay before the first online episode")
    online.add_argument("--expert-condition-vla", action="store_true",
                        help="Condition demonstration actions on their frozen VLA references")

    evaluate = commands.add_parser("eval-tianji", help="Evaluate base VLA or an RLT learner on paired seeds")
    evaluate.add_argument("--learner", type=Path)
    evaluate.add_argument("--episodes", type=int, default=50)
    evaluate.add_argument("--out", type=Path, help="Write per-seed results and summary to a new JSON file")

    for command in (online, evaluate):
        backbone_args(command)
        command.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
        command.add_argument("--token", type=Path, required=command is online)
        command.add_argument("--device", default="cpu")
        command.add_argument("--shared-prefix", action="store_true",
                             help="Reuse the native π0.5 action sampler's prefix computation")
        command.add_argument("--max-steps", type=int)
        phase_group = command.add_mutually_exclusive_group()
        phase_group.add_argument("--critical-start-step", type=int)
        phase_group.add_argument("--critical-phase", choices=("ready",),
                                 help="Begin RLT when every module's native slot timer is ready")
        command.add_argument("--vla-seed", type=int, help="Reset OpenPI sampling to this seed each episode")
        command.add_argument("--chunk-size", type=int, default=10)
        command.add_argument("--base-chunk-size", type=int, default=20)
        command.add_argument("--stride", type=int, default=2)
        command.add_argument("--gamma", type=float, default=0.99)
        command.set_defaults(run=tianji_run)

    evaluate.set_defaults(seed=21001)
    args = parser.parse_args()
    for field in ("torch_threads", "episodes", "eval_episodes", "checkpoint_every", "expert_stride", "warmup_chunk_size"):
        value = getattr(args, field, None)
        if value is not None and value < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if min(getattr(args, "expert_demos", 0), getattr(args, "expert_bootstrap_updates", 0)) < 0:
        parser.error("Expert demo count and initial update count must be nonnegative")
    if getattr(args, "resume_every", 0) < 0 or (getattr(args, "resume", False) and not args.resume_every):
        parser.error("Use a positive --resume-every with --resume")
    if getattr(args, "expert_demos", 0) and args.critical_start_step is not None:
        parser.error("Expert replay supports whole episodes and the native READY phase")
    if getattr(args, "expert_condition_vla", False) and not args.expert_demos:
        parser.error("--expert-condition-vla requires --expert-demos")
    if min(args.seed, getattr(args, "critical_start_step", None) or 0, getattr(args, "vla_seed", None) or 0) < 0:
        parser.error("Seeds and critical phase start must be nonnegative")
    torch.set_num_threads(args.torch_threads)
    args.run(args)


if __name__ == "__main__":
    main()
