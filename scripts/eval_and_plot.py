# --- NumPy 2.x -> 1.x pickle shim: must be the first thing in this file ---
import sys, importlib
try:
    import numpy._core  # already in NumPy 2.x, nothing to do
except Exception:
    import numpy.core as _ncore
    sys.modules['numpy._core'] = _ncore
    for _n in ('numeric','fromnumeric','_methods','multiarray','umath','overrides','shape_base'):
        try:
            sys.modules[f'numpy._core.{_n}'] = importlib.import_module(f'numpy.core.{_n}')
        except Exception:
            pass
# --------------------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC

def evaluate_and_save(model_path: str, episodes: int, out_csv: Path) -> np.ndarray:
    """评估一个模型，返回每回合回报并保存为 CSV。"""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("Walker2d-v4", render_mode=None)  # 无渲染更快
    model = SAC.load(
    model_path,
    env=env,
    custom_objects={
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        "lr_schedule": (lambda _: 3e-4), 
    },
)

    returns = []
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = truncated = False
        ep_ret = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
        returns.append(ep_ret)

    env.close()

    arr = np.asarray(returns, dtype=float)
    np.savetxt(out_csv, arr, delimiter=",", header="episode_return", comments="")
    print(f"[OK] Saved {episodes} returns -> {out_csv}")
    print(f"     mean={arr.mean():.2f}, std={arr.std():.2f}")
    return arr

def moving_average(y: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return y
    k = int(k)
    k = min(k, max(1, len(y)//5))  # 防止窗口过大
    w = np.ones(k, dtype=float) / k
    pad = k // 2
    ypad = np.pad(y, (pad, k-1-pad), mode="edge")
    return np.convolve(ypad, w, mode="valid")

def plot_curves(csv_files, labels, out_png: Path, smooth: int):
    plt.figure(figsize=(9, 5))
    for fp, lab in zip(csv_files, labels):
        y = np.loadtxt(fp, delimiter=",", skiprows=1)
        x = np.arange(1, len(y)+1)
        if smooth and smooth > 1:
            ys = moving_average(y, smooth)
            plt.plot(x, ys, label=f"{lab} (MA{smooth})", linewidth=2)
        # 原始曲线淡一些
        plt.plot(x, y, alpha=0.25, linewidth=1)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Walker2D: Episode Returns")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved figure -> {out_png}")

def main():
    ap = argparse.ArgumentParser(description="评估多个 Walker2D 模型并画对比图")
    ap.add_argument("--models", nargs="+", required=True,
                    help="一个或多个模型路径（.zip）")
    ap.add_argument("--labels", nargs="+", help="与 models 对应的曲线名称")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--csv_dir", default="outputs/logs", help="CSV 输出目录")
    ap.add_argument("--out_png", default="outputs/figures/walker2d_returns.png")
    ap.add_argument("--smooth", type=int, default=25, help="移动平均窗口；1=关闭")
    args = ap.parse_args()

    # 准备标签
    if args.labels is None:
        labels = [Path(p).stem for p in args.models]
    else:
        assert len(args.labels) == len(args.models), "labels 数量要与 models 一致"
        labels = args.labels

    csv_paths = []
    for m, lab in zip(args.models, labels):
        out_csv = Path(args.csv_dir) / f"{lab}_returns.csv"
        evaluate_and_save(m, args.episodes, out_csv)
        csv_paths.append(out_csv)

    plot_curves(csv_paths, labels, Path(args.out_png), args.smooth)

if __name__ == "__main__":
    main()