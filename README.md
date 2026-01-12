# Reset-Efficient Reinforcement Learning for Robotics

Systematic comparison on Walker2d-v4: Free baseline, Reset-cost punishment, Dual-policy recovery, and Goal-conditioned skills.

**Key result:** Dual-policy recovery reduces catastrophic resets by ~65% while retaining ~90% of baseline return.

## Features

- Four strategies: Free baseline, Reset-cost punishment, Dual-policy recovery, Goal-conditioned skills
- Unified evaluation: 1000-episode returns and 100-episode gait quality (posture / cadence / vertical stability / effort)
- Clean, modular code + YAML configs + CLI scripts → one-command reproducibility
- Poster-ready high-res snapshots & optional video rendering

## Project Structure

```
reset-efficient-rl/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ configs/
│  ├─ exp1_free.yaml
│  ├─ exp2_punish.yaml
│  ├─ exp3_recovery.yaml
│  └─ exp4_goal.yaml
├─ src/
│  ├─ __init__.py
│  ├─ common.py            # callbacks, utilities
│  ├─ cost_wrapper.py      # Exp2: penalty on catastrophic reset
│  ├─ goal_eval_wrapper.py # eval wrapper for 19-D goal-conditioned models
│  ├─ train_loops.py       # training loops for Exp1/2/3/4
│  └─ metrics_gait.py      # gait metrics computation
├─ scripts/
│  ├─ train.py
│  ├─ eval_returns.py
│  ├─ eval_and_plot.py
│  ├─ eval_gait.py
│  ├─ plot_gait_compare.py
│  ├─ make_gui_shots.py
│  └─ render_clips.py
└─ outputs/                # generated artifacts (git-ignored)
   ├─ logs/
   ├─ models/
   ├─ gait_logs/
   └─ figures/
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

If GUI rendering fails, use off-screen tools (`make_gui_shots.py`, `render_clips.py`) which rely on `render_mode="rgb_array"`.

## Configuration Examples

### configs/exp2_punish.yaml

```yaml
env_id: Walker2d-v4
seed: 0
n_envs: 46
algo: SAC
total_timesteps: 1000000

wrapper: cost
wrapper_kwargs:
  cost: -1000.0

buffer_size: 1000000
batch_size: 8192
learning_starts: 10000
gamma: 0.99
tau: 0.005
learning_rate: 3e-4

log_dir: outputs/logs/exp2
save_path: outputs/models/walker2d_sac_exp2.zip
```

### configs/exp3_recovery.yaml

```yaml
env_id: Walker2d-v4
seed: 0
n_envs: 1
algo: SAC
total_timesteps: 1000000

trigger:
  low_h: 0.84
  high_h: 1.92
  angle: 0.96
recovery:
  success_dist: 0.5
  max_steps: 1000

buffer_size: 1000000
batch_size: 8192
learning_starts: 10000
gamma: 0.99
tau: 0.005
learning_rate: 3e-4

log_dir: outputs/logs/exp3
save_path: outputs/models/walker2d_sac_exp3.zip
```

## Quickstart (Reproduce)

### 1) Train

```bash
# Exp1: Free baseline
python scripts/train.py --config configs/exp1_free.yaml

# Exp2: Reset-cost punishment
python scripts/train.py --config configs/exp2_punish.yaml

# Exp3: Dual-policy recovery (Forward + Recovery)
python scripts/train.py --config configs/exp3_recovery.yaml

# Exp4: Goal-conditioned (19-D obs = 17 state + 2 goal one-hot)
python scripts/train.py --config configs/exp4_goal.yaml
```

### 2) Evaluate 1000-episode returns & plot

```bash
python scripts/eval_and_plot.py \
  --models outputs/models/walker2d_sac_exp1.zip \
          outputs/models/walker2d_sac_exp2.zip \
          outputs/models/walker2d_sac_exp3.zip \
          outputs/models/walker2d_goal_conditioned.zip \
  --labels exp1 exp2 exp3 goal \
  --episodes 1000 \
  --csv_dir outputs/logs \
  --out_png outputs/figures/episode_returns.png \
  --smooth 25
```

### 3) Gait quality (100 episodes) & figures

```bash
# Produce per-step parquet + per-episode CSV; auto-detect 19-D goal-conditioned models
python scripts/eval_gait.py \
  --models outputs/models/walker2d_sac_exp1.zip \
          outputs/models/walker2d_sac_exp2.zip \
          outputs/models/walker2d_sac_exp3.zip \
          outputs/models/walker2d_goal_conditioned.zip \
  --labels exp1 exp2 exp3 goal \
  --episodes 100 \
  --out_dir outputs/gait_logs

# Plot radar + box plots
python scripts/plot_gait_compare.py \
  --in_dir outputs/gait_logs \
  --out_dir outputs/figures
```

### 4) Poster-ready snapshots / optional video

```bash
# High-res multi-camera contact sheet
python scripts/make_gui_shots.py \
  --model_path outputs/models/walker2d_sac_exp3.zip \
  --steps 260 300 340 \
  --cams side iso front bird \
  --out_dir outputs/figures \
  --prefix exp3 \
  --width 1600 --height 900 --supersample 2 \
  --target_w 1600 --target_h 900 \
  --deterministic \
  --contact_sheet --sheet_cols 4

# Short clip (optional)
python scripts/render_clips.py \
  --model outputs/models/walker2d_sac_exp3.zip \
  --out outputs/videos/exp3.mp4
```

## Main Results (our runs)

| Experiment | Episode Return (1000 eps) | Catastrophic Resets (train cumulative) | Style |
|------------|---------------------------|----------------------------------------|-------|
| Exp1 (Free) | ~4700 ± 170 | 10,749 | Fast & aggressive |
| Exp2 (Punish) | ~2900 ± 280 | 6,501 | Safe but over-conservative |
| Exp3 (Recovery) | ~4200 ± 330 | 3,720 | −65% resets, ~90% return |
| Exp4 (Goal) | ~3880 ± 430 | 9,749 | Simple deployment |

- Returns use deterministic evaluation.
- Gait metrics use 100 episodes; return comparison uses 1000 episodes.

## Gait Metrics (meaning)

- **pitch_abs_mean** — mean absolute torso pitch (lower = more upright/stable)
- **z_std** — std of torso height (lower = smoother CoM trajectory)
- **freq_pitch** — dominant frequency of pitch oscillation (cadence proxy; higher = stronger rhythm)
- **act_norm_mean** — mean L2 norm of actions (lower = lower control effort)

### Observed patterns

- **Recovery (Exp3)** learns confident, dynamic gait (higher cadence) while keeping resets low via recovery → high returns.
- **Punish (Exp2)** becomes stiff/over-cautious; resets drop but returns degrade.
- **Free / Goal** are balanced but less optimal trade-offs.

## Notes

- **Goal-conditioned models (Exp4)** expand observation to 19-D by appending a 2-D goal one-hot. Our eval scripts auto-wrap the env with `GoalEvalWrapper` and fix the goal to `[1, 0]` ("forward mode") for fair comparison.
- **Parallel envs:** Exp3 uses `n_envs=1` (serial state machine). Exp1/2/4 use `n_envs=46` for faster sampling.
- **NumPy 2.x users:** scripts include a minimal compatibility shim for older SB3 pickles.

## Reproducibility Checklist

- Fixed seeds (`--seed` in configs)
- Deterministic evaluation (`deterministic=True`)
- All hyperparameters and thresholds in `configs/*.yaml`
- All generated artifacts saved under `outputs/` with clear names

## Troubleshooting

### Shape mismatch (17 vs 19) when loading
You are loading a goal-conditioned model with a 17-D env. Use our eval scripts (auto-wrap) or manually wrap `GoalEvalWrapper`.

### GUI overlay crash / OpenGL issues
Use off-screen tools (`make_gui_shots.py`, `render_clips.py`) which call `render_mode="rgb_array"`.

## License

MIT License © 2025

## Acknowledgements

Built with Gymnasium, MuJoCo, Stable-Baselines3, PyTorch.

## Collaborators

Qinrui Deng, Xiaotong Yan, Zhaoyan Fan, Zhuojie Wu
