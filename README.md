# Reset-Efficient Reinforcement Learning for Robotics

Systematic comparison on Walker2d-v4: Free baseline, Reset-cost punishment, and Dual-policy recovery.

**Key result:** Dual-policy recovery reduces catastrophic resets by ~65% while retaining ~90% of baseline return.

## Features

- Three experimental strategies: Free baseline, Reset-cost punishment, Dual-policy recovery
- Unified evaluation framework for episode returns and reset statistics
- Clean, modular code with CLI interface
- Comprehensive visualization tools for results analysis

## Project Structure

```
Trained_eval/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── common.py              # Callbacks and utilities
│   └── cost_wrapper.py        # Environment wrapper for reset-cost punishment
├── scripts/
│   ├── train.py               # Training script for all experiments
│   ├── eval.py                # Model evaluation
│   ├── eval_and_plot.py       # Batch evaluation with plotting
│   ├── make_gui_shots.py      # Generate screenshots
│   ├── plot_bars.py           # Bar chart visualization
│   └── plot_reward_resets_vs_steps.py  # Trajectory analysis
├── outputs/                   # Generated artifacts (git-ignored)
│   ├── logs/                  # TensorBoard logs
│   ├── models/                # Trained models
│   └── figures/               # Generated plots
└── training_records/          # Training notes and records
```

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python3 -c "import gymnasium; import stable_baselines3; print('✓ Installation successful')"
```

## Quick Start

### Train Models

```bash
# Experiment 1: Free baseline (no constraints)
python scripts/train.py --experiment 1 --steps 1000000

# Experiment 2: Reset-cost punishment (-1000 penalty on falls)
python scripts/train.py --experiment 2 --steps 1000000

# Experiment 3: Dual-policy recovery (Forward + Recovery policies)
python scripts/train.py --experiment 3 --steps 1000000
```

**Training Options:**
- `--experiment`: Experiment number (1, 2, or 3)
- `--steps`: Total training timesteps (default: 1,000,000)
- `--log_dir`: TensorBoard log directory (default: `outputs/logs`)
- `--save_path`: Model save path (default: `outputs/models/walker2d_sac`)

**Output:**
- Models: `outputs/models/walker2d_sac_exp{1,2,3}.zip`
- Logs: `outputs/logs/exp{1,2,3}/`

### Evaluate Models

#### Single Model Evaluation

```bash
python scripts/eval.py \
  --model_path outputs/models/walker2d_sac_exp1.zip \
  --episodes 20
```

#### Batch Evaluation with Plotting

```bash
python scripts/eval_and_plot.py \
  --models outputs/models/walker2d_sac_exp1.zip \
          outputs/models/walker2d_sac_exp2.zip \
          outputs/models/walker2d_sac_exp3.zip \
  --labels exp1 exp2 exp3 \
  --episodes 1000 \
  --csv_dir outputs/logs \
  --out_png outputs/figures/episode_returns.png \
  --smooth 25
```

**Output:**
- CSV files: `outputs/logs/exp{1,2,3}_returns.csv`
- Plot: `outputs/figures/episode_returns.png`

### Visualization

#### Generate Bar Charts

```bash
python scripts/plot_bars.py
```

Edit the script to update experimental results before running.

**Output:** `outputs/figures/bar_*.png`

#### Generate Trajectory Plots

```bash
python scripts/plot_reward_resets_vs_steps.py \
  --models outputs/models/walker2d_sac_exp1.zip \
          outputs/models/walker2d_sac_exp2.zip \
          outputs/models/walker2d_sac_exp3.zip \
  --labels exp1 exp2 exp3 \
  --steps 200000 \
  --smooth 200 \
  --out_png outputs/figures/reward_resets_vs_steps.png
```

#### Generate Screenshots

```bash
python scripts/make_gui_shots.py \
  --model_path outputs/models/walker2d_sac_exp3.zip \
  --steps 200 \
  --shot_steps 50 150 \
  --out outputs/figures/shots1.png outputs/figures/shots2.png
```

## Experimental Details

### Experiment 1: Free Baseline

**Description:** Standard SAC training without any reset penalties.

**Configuration:**
- Environment: Walker2d-v4
- Parallel environments: 24
- Algorithm: SAC
- Total timesteps: 1,000,000
- No environment wrapper

**Command:**
```bash
python scripts/train.py --experiment 1 --steps 1000000
```

### Experiment 2: Reset-Cost Punishment

**Description:** Penalize catastrophic resets (falls) with -1000 reward.

**Configuration:**
- Environment: Walker2d-v4 + CostPunishWrapper
- Parallel environments: 24
- Reset penalty: -1000
- Other parameters: same as Exp1

**Command:**
```bash
python scripts/train.py --experiment 2 --steps 1000000
```

### Experiment 3: Dual-Policy Recovery

**Description:** Train two policies - a forward policy and a recovery policy to prevent falls.

**Configuration:**
- Environment: Walker2d-v4
- Serial execution: n_envs=1 (required for state machine)
- Two SAC policies: Forward (Policy_F) and Recovery (Policy_R)
- Trigger thresholds:
  - `low_height`: 0.84 (vs. failure at 0.8)
  - `high_height`: 1.92 (vs. failure at 2.0)
  - `angle`: 0.96 (vs. failure at 1.0)
- Recovery success threshold: 0.5
- Maximum recovery steps: 1000

**Command:**
```bash
python scripts/train.py --experiment 3 --steps 1000000
```

## Main Results

| Experiment | Episode Return (1000 eps) | Catastrophic Resets (train cumulative) | Style |
|------------|---------------------------|----------------------------------------|-------|
| Exp1 (Free) | ~4700 ± 170 | 10,749 | Fast & aggressive |
| Exp2 (Punish) | ~2900 ± 280 | 6,501 | Safe but over-conservative |
| Exp3 (Recovery) | ~4200 ± 330 | 3,720 | **−65% resets, ~90% return** |

*Returns use deterministic evaluation over 1000 episodes.*

### Key Findings

- **Recovery (Exp3)** achieves the best trade-off: high returns with significantly reduced resets
- **Punishment (Exp2)** reduces resets but becomes overly conservative, sacrificing performance
- **Free baseline (Exp1)** achieves highest returns but with most frequent falls

## Algorithm Details

### SAC Hyperparameters

All experiments use the following SAC configuration:

```python
buffer_size: 1,000,000
batch_size: 8,192
learning_starts: 10,000
gamma: 0.99
tau: 0.005
learning_rate: 3e-4
train_freq: 1
```

### Experiment-Specific Settings

**Exp1 & Exp2:**
- Parallel environments: 24
- Standard SB3 training loop

**Exp3:**
- Serial execution: 1 environment
- Custom training loop with state machine
- Two separate replay buffers
- Alternating policy updates

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

Then open http://localhost:6006 in your browser.

### Logged Metrics

- `train/policy_F_reward`: Forward policy episode return
- `train_stats/cumulative_expensive_resets`: Total catastrophic resets
- `train_stats/cumulative_artificial_resets`: Total timeout resets
- `train_stats/cumulative_recovery_success`: Successful recoveries (Exp3 only)
- `train_stats/triggers_safe`: Safety-triggered recoveries (Exp3 only)
- `train_stats/triggers_time`: Timeout-triggered recoveries (Exp3 only)

## Troubleshooting

### ModuleNotFoundError: No module named 'gymnasium'

```bash
# Activate virtual environment first
source .venv/bin/activate

# Then install dependencies
pip install -r requirements.txt
```

### Model Loading Errors

If you encounter pickle/numpy compatibility issues:

```bash
# The scripts include NumPy 2.x compatibility shims
# Ensure you're using the provided eval scripts
python scripts/eval.py --model_path <your_model.zip>
```

### GUI Rendering Issues on macOS

Use off-screen rendering mode:

```bash
python scripts/make_gui_shots.py --model_path <model.zip>
```

This uses `render_mode="rgb_array"` which avoids OpenGL issues.

## File Organization

### Important Files

- **Core modules:** `src/common.py`, `src/cost_wrapper.py`
- **Training:** `scripts/train.py`
- **Evaluation:** `scripts/eval.py`, `scripts/eval_and_plot.py`
- **Visualization:** `scripts/plot_*.py`, `scripts/make_gui_shots.py`

### Generated Outputs

All training outputs are saved to `outputs/`:

```
outputs/
├── logs/
│   ├── exp1/
│   │   ├── SAC_*/          # TensorBoard logs
│   │   └── exp1_returns.csv
│   ├── exp2/
│   └── exp3/
├── models/
│   ├── walker2d_sac_exp1.zip
│   ├── walker2d_sac_exp2.zip
│   └── walker2d_sac_exp3.zip
└── figures/
    ├── episode_returns.png
    ├── bar_avg_return.png
    ├── bar_expensive_resets.png
    ├── bar_timelimit_resets.png
    └── reward_resets_vs_steps.png
```

## Dependencies

Core dependencies (see `requirements.txt` for complete list):

- `gymnasium[mujoco]` - RL environment
- `stable-baselines3[extra]` - RL algorithms
- `torch` - Deep learning backend
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `tensorboard` - Training monitoring
- `imageio` - Image processing

## Reproducibility

### Ensure Reproducibility

1. **Fixed seeds:** Set via `seed=0` in code
2. **Deterministic evaluation:** Use `deterministic=True` in eval scripts
3. **Record hyperparameters:** All settings documented in code comments
4. **Save artifacts:** Models and logs saved with clear naming

### Re-run Experiments

```bash
# Clean previous outputs
rm -rf outputs/logs/* outputs/models/*

# Run all experiments
for exp in 1 2 3; do
    python scripts/train.py --experiment $exp --steps 1000000
done

# Evaluate all models
python scripts/eval_and_plot.py \
  --models outputs/models/walker2d_sac_exp{1,2,3}.zip \
  --labels exp1 exp2 exp3 \
  --episodes 1000
```

## License

MIT License © 2025

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{walker2d_reset_efficient_2025,
  title={Reset-Efficient Reinforcement Learning for Robotics},
  author={Tingting Xu and Collaborators},
  year={2025},
  url={https://github.com/tingting-xu824/rl_project}
}
```

## Acknowledgements

Built with:
- [Gymnasium](https://gymnasium.farama.org/) - RL environments
- [MuJoCo](https://mujoco.org/) - Physics simulation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [PyTorch](https://pytorch.org/) - Deep learning

## Collaborators

Qinrui Deng, Xiaotong Yan, Zhaoyan Fan, Zhuojie Wu
