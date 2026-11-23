"""Run a held-out, paired Tianji insert_one experiment with one frozen π0.5."""

import argparse
from copy import copy
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
from statistics import mean
import time

import torch

from .online import Learner, OnlineConfig
from .openpi import FrozenOpenPI
from .rollout import OnlineTrainer, RolloutConfig, run_episode
from .tianji import TianjiEnv, ready_phase
from .token import load_token


def _stats(rows):
    successful_steps = [row["steps"] for row in rows if row["success"]]
    return {
        "episodes": len(rows),
        "successes": len(successful_steps),
        "success_rate": len(successful_steps) / len(rows),
        "mean_steps_all": mean(row["steps"] for row in rows),
        "mean_steps_success_only": mean(successful_steps) if successful_steps else None,
    }


def _collect(env, vla, token, learner, rollout, args, shared, log):
    trainer = OnlineTrainer(
        learner, seed=args.train_seed, warmup_steps=10**9,
        updates_per_transition=args.utd, asynchronous=True,
    )
    replay_vla = copy(vla)
    train_rows = []
    try:
        for index in range(args.warmup_episodes + args.online_episodes):
            seed = args.train_seed + index
            warmup = index < args.warmup_episodes
            result = trainer.train_episode(
                env, vla, token, rollout, seed, replay_vla=replay_vla, **shared,
            )
            trainer.finish()
            row = {"split": "warmup" if warmup else "online", "seed": seed, **asdict(result)}
            train_rows.append(row)
            log({
                **row, "total_transitions": trainer.transitions, "seconds": time.monotonic() - args.start,
                "critic_updates": learner.critic_updates, "actor_updates": learner.actor_updates,
                **trainer.metrics,
            })
            if index + 1 == args.warmup_episodes:
                if not len(trainer.replay):
                    raise RuntimeError("No warmup transition reached the critical phase")
                updates = min(args.prefit_cap, args.utd * len(trainer.replay))
                for _ in range(updates):
                    trainer.metrics.update(learner.update(trainer.replay.sample(learner.config.batch_size)))
                trainer.warmup_steps = 0
                log({"event": "warmup_replay_fit", "critic_updates": updates, **trainer.metrics})
            if not warmup or index + 1 == args.warmup_episodes:
                target = args.out / "learner.pt"
                temp = target.with_suffix(".tmp")
                torch.save({
                    "learner": learner.state_dict(), "rollout": asdict(rollout),
                    "critical_start_step": 0, "critical_phase": "ready",
                    "vla_seed": args.vla_seed, "vla_checkpoint": str(args.vla_checkpoint.resolve()),
                    "level": "insert_one", "training_episodes": index + 1,
                }, temp)
                temp.replace(target)
    finally:
        trainer.close()
    return train_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--token", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=31001)
    parser.add_argument("--test-seed", type=int, default=21001)
    parser.add_argument("--test-count", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument("--online-episodes", type=int, default=8)
    parser.add_argument("--utd", type=int, default=5)
    parser.add_argument("--prefit-cap", type=int, default=2500)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--vla-seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.start = time.monotonic()
    if min(args.test_count, args.warmup_episodes, args.online_episodes, args.utd, args.prefit_cap,
           args.torch_threads) < 1:
        parser.error("Episode counts, update counts and thread count must be positive")
    test_seeds = list(range(args.test_seed, args.test_seed + args.test_count))
    train_seeds = list(range(args.train_seed, args.train_seed + args.warmup_episodes + args.online_episodes))
    if set(test_seeds) & set(train_seeds):
        parser.error("Training and test seeds must be disjoint")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error(f"Select a fresh result directory: {args.out}")

    def log(record):
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with (args.out / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.train_seed)
    rollout = RolloutConfig(stride=args.stride, base_chunk_size=50)
    settings = OnlineConfig(beta=0.5, action_std=0.01, batch_size=128)
    config = {
        "level": "insert_one", "critical_phase": "ready", "vla_seed": args.vla_seed,
        "train_seeds": train_seeds, "test_seeds": test_seeds, "warmup_episodes": args.warmup_episodes,
        "utd": args.utd, "prefit_cap": args.prefit_cap, "rollout": asdict(rollout),
        "learner": asdict(settings), "vla_checkpoint": str(args.vla_checkpoint.resolve()),
        "token": str(args.token.resolve()), "max_steps": 480, "control_hz": 30,
    }
    (args.out / "experiment.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    log({"event": "start", **config})

    token = load_token(args.token, args.device)
    vla = FrozenOpenPI.from_tianji(args.vla_checkpoint)
    with TianjiEnv("insert_one") as env:
        learner = Learner(
            token.config.token_dim + env.proprio_dim, rollout.chunk_size, env.action_dim,
            settings, device=args.device, action_projection=env.action_projection(args.device),
        )
        shared = {"phase": ready_phase, "action_offset": env.action_offset, "vla_seed": args.vla_seed}
        train = _collect(env, vla, token, learner, rollout, args, shared, log)
        deterministic = partial(learner.act, deterministic=True)
        pairs = []
        for seed in test_seeds:
            record = {"seed": seed}
            for name, policy in (("base", None), ("rlt", deterministic)):
                result = run_episode(
                    env, vla, token if policy is not None else None, rollout, seed,
                    policy=policy, **shared,
                )
                record[name] = asdict(result)
                log({"split": "test", "policy": name, "seed": seed, **record[name],
                     "seconds": time.monotonic() - args.start})
            pairs.append(record)

    both_pass = [row for row in pairs if row["base"]["success"] and row["rlt"]["success"]]
    summary = {
        "config": config, "training": train, "train_updates": {
            "critic": learner.critic_updates, "actor": learner.actor_updates,
        },
        "base": _stats([row["base"] for row in pairs]),
        "rlt": _stats([row["rlt"] for row in pairs]),
        "both_succeeded": len(both_pass),
        "mean_paired_steps_change_both_succeeded": (
            mean(row["rlt"]["steps"] - row["base"]["steps"] for row in both_pass) if both_pass else None
        ),
        "pairs": pairs, "elapsed_seconds": time.monotonic() - args.start,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log({"event": "complete", **{key: summary[key] for key in ("base", "rlt", "both_succeeded", "train_updates")}})


if __name__ == "__main__":
    main()
