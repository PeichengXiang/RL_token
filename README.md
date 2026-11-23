# RL_token

Physical Intelligence 的 **RL Token** 独立复现，包含 PyTorch 实现、OpenPI π0/π0.5 适配器和天机光模块插拔仿真接口。CPU 示例可独立跑通两阶段训练和配对评估。 [论文](https://arxiv.org/html/2604.23073v2) · [作者项目页](https://www.pi.website/research/rlt)

| 阶段 | 训练内容 | 主要产物 |
| --- | --- | --- |
| 1. RL token | 冻结 VLA，在最后一层特征序列后加可学习读出 token。因果解码器在 teacher forcing 下重建原始特征。 | `token.pt` |
| 2. 在线 RL | 冻结 VLA 和 token 编码器。Actor 用 RL token、本体状态及 VLA 参考动作生成完整动作块，双 critic 用 replay 更新。 | `learner.pt`、`summary.json` |

## 快速开始

需要 Python 3.10+。这一节只需要 NumPy 和 PyTorch。

```bash
git clone https://github.com/PeichengXiang/RL_token.git
cd RL_token
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m unittest discover -s tests -v
python -m rlt.cli --torch-threads 2 demo --sync \
  --token-steps 30 --warmup-steps 32 --episodes 12 --eval-episodes 5 --out runs/toy_smoke
```

示例会写出两阶段权重和 `summary.json`，并在同一组留出种子上记录基础策略和 RLT 的成功率、控制步数。快速示例的预算用于检查运行链路。OpenPI 集成测试在缺少 JAX 或 OpenPI 时跳过。

## 天机光模块插拔

需要另行准备 [天机 benchmark](https://github.com/PeichengXiang/insert_benchmark)、Isaac Sim 5.1、Pi_05 适配器、OpenPI JAX 运行环境以及天机 π0.5 权重。权重目录须包含 `params/`、`training_config.json` 和 `assets/tianji_optical_modules/norm_stats.json`。天机 CLI 默认按以下相邻目录查找 Pi_05 和 XPolicyLab。

```text
workspace/
├── insert_benchmark/
│   ├── benchmark_data/
│   └── third_party/XPolicyLab/
└── policy/
    ├── Pi_05/
    └── RL_token/
```

```bash
python -m pip install -e '.[tianji-data]'

# 在 workspace/policy/RL_token 内设置实际路径
BENCHMARK="$(cd ../../insert_benchmark && pwd)"
PI_PY=/path/to/configured/openpi-python
VLA_CHECKPOINT=/path/to/tianji-pi05-checkpoint
```

`PI_PY` 应使用已配置 JAX、OpenPI、PyTorch、NumPy 和 h5py 的解释器，或指向能启动该环境的 Pi_05 launcher。按 benchmark 文档准备仿真资产，可用 `ISAAC_SIM_ROOT` 指定 Isaac Sim 的安装目录。

### 第一阶段

每个任务独立构建特征缓存并训练 token。第二阶段使用与缓存来源一致的 π0.5 checkpoint 和语言掩码设置。

```bash
"$PI_PY" -m rlt.cli cache-tianji \
  --dataset "$BENCHMARK/benchmark_data" --level insert_one \
  --vla-checkpoint "$VLA_CHECKPOINT" --frame-stride 60 \
  --out runs/insert_one_features

python -m rlt.cli train-token \
  --cache runs/insert_one_features --steps 10000 --batch-size 2 \
  --device cuda --out runs/insert_one_token
```

`--device cuda` 需要 CUDA 版 PyTorch。可用 `--max-episodes` 和 `--max-frames` 做少量特征的接口验证。默认 2,048 维 token 的 checkpoint 约 801 MiB。

### 第二阶段与续训

以下命令按完整回合训练 `insert_one`。对 `timed_single` 换用该任务的 token 和独立输出目录，并加上 `--critical-phase ready`，在时钟到 READY 时交棒。

```bash
JAX_PLATFORMS=cuda "$PI_PY" -m rlt.cli --torch-threads 4 train-tianji \
  --benchmark "$BENCHMARK" --level insert_one --vla-checkpoint "$VLA_CHECKPOINT" \
  --token runs/insert_one_token/token.pt --out runs/insert_one_stage2 \
  --episodes 400 --seed 32001 --vla-seed 0 --warmup-steps 1000 \
  --warmup-chunk-size 50 --base-chunk-size 50 --chunk-size 10 --stride 2 --utd 5 \
  --shared-prefix --separate-replay-rng --device cpu \
  --checkpoint-every 1 --resume-every 10
```

- `--warmup-chunk-size 50` 使用原生 π0.5 的 50 步动作块采集 warm-up，actor 仍按 10 步动作块执行。
- 可用 `--expert-dataset`、`--expert-demos` 和 `--expert-bootstrap-updates` 从训练集成功示范预填 replay。启用 `--expert-condition-vla` 时，actor 以 π0.5 参考动作作输入，以专家执行的动作作行为克隆目标。示范预热更新单独计数。
- `learner.pt` 用于评估，也可用于 `--init-learner` 热启动。启用 `--resume-every` 后，重跑同一训练配置和输出目录并加 `--resume`，从 `training.pt` 恢复回放、优化器、随机数状态及已提交的回合计数。
- `scripts/pai_stage1.py` 可打包冻结特征并在 CUDA worker 上训练 token。 `scripts/pai_cache_features.py` 可通过显式的 `--openpi`、`--trainer` 和 `--dataset` 接入已有的 Pi_05 云端训练目录。

### 同种子评估

```bash
eval_args=(
  --benchmark "$BENCHMARK" --level insert_one --vla-checkpoint "$VLA_CHECKPOINT"
  --seed 21001 --episodes 20 --vla-seed 0 --chunk-size 10 --base-chunk-size 50
)
"$PI_PY" -m rlt.cli eval-tianji "${eval_args[@]}" --out runs/insert_one_base.json
"$PI_PY" -m rlt.cli eval-tianji "${eval_args[@]}" \
  --token runs/insert_one_token/token.pt --learner runs/insert_one_stage2/learner.pt \
  --out runs/insert_one_rlt.json
```

原生测试种子为 `insert_one` 的 21001–21020 和 `timed_single` 的 25001–25020。训练时避开测试种子。两组使用相同的 seed、步数上限、动作块长度和交棒条件。`--learner` 会采用 checkpoint 内的 rollout 配置，基础策略需显式传入对应的动作块配置。评估 `timed_single` 的 READY 策略时，给基础策略也指定 `--critical-phase ready`。结果包含成功率、全体回合平均控制步数、成功回合平均控制步数和逐回合数据。

## 复现口径

| 项目 | 本实现 |
| --- | --- |
| VLA | 实现论文允许的 α=0 路径，Stage 1 读取已微调并冻结的 π0.5。论文真机实验使用 π0.6。 |
| Token 编解码器 | 两边各 2 层、8 个注意力头、FFN 为宽度的 4 倍。默认读出维度 2,048，AdamW 学习率 3e-4。架构细节和优化参数是本实现选值。 |
| Actor 与 Critic | Actor 输出固定标准差的高斯动作块。双 critic 取最小目标值，actor 使用 Q 与行为克隆约束。训练时以 50% 概率遮蔽参考动作输入。 |
| 时序与奖励 | Actor 块长 10，回放起点间隔 2 个控制步。达到 warm-up 和 batch 条件后，每条新增 transition 默认更新 critic 5 次，每 2 次 critic 更新做 1 次 actor 更新。天机以 30 Hz 运行，由原生完整成功判定给出稀疏 +1。 |
| 天机动作 | 双臂关节动作以块起点为参考表示相对位置，夹爪使用绝对开度，执行前换算为 16 维绝对目标。论文真机实验是 50 Hz、14 维相对动作，由操作员提供二元成功信号。 |
| 训练预算 | 论文附录 B 给出第一阶段 2,000–10,000 次梯度更新，§VI-B 给出第二阶段每任务 400–1,000 个在线回合。上面的命令示例分别取 10,000 步和 400 回合。 |

论文未公布第一阶段的重建误差曲线或验证达标线。合成示例用于验证实现和运行链路，任务表现通过单独的同种子评估测量。

仓库提供代码、测试、配置和运行示例。权重、示范数据、仿真资产、特征缓存及训练产物由各自的外部环境提供。
