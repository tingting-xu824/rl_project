# plot_reward_resets_vs_steps.py
# 生成：随 steps 变化的 reward（移动平均，左轴）+ 累计 resets（右轴，虚线）
# 兼容当前的 sb3 / numpy 组合，并尽量避免反序列化问题

# --- robust NumPy2<->NumPy1 shim: must be the first lines ---
import sys, importlib, types
try:
    import numpy._core  # numpy>=2 已自带该命名空间
except Exception:
    import numpy as _np
    import numpy.core as _ncore
    # 挂 alias: numpy._core -> numpy.core
    sys.modules['numpy._core'] = _ncore
    for _name in ('numeric','fromnumeric','_methods','multiarray','umath','overrides','shape_base'):
        try:
            _m = importlib.import_module(f'numpy.core.{_name}')
            sys.modules[f'numpy._core.{_name}'] = _m
        except Exception:
            pass
    # 兼容旧 pickle 里用到的 numpy._core.numeric._frombuffer
    try:
        num_mod = sys.modules['numpy._core.numeric']
        if not hasattr(num_mod, '_frombuffer'):
            num_mod._frombuffer = _np.frombuffer
    except Exception:
        pass
# ----------------------------------------------------------------

import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import SAC

def moving_average(x, k):
    if k <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    # 简单滑窗均值
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / float(k)
    # 前 k-1 用逐步均值补齐，避免长度对不上
    head = [np.mean(x[:i+1]) for i in range(min(k-1, len(x)))]
    return np.array(head + list(y), dtype=float)

def run_trajectory(model_path, total_steps=200000, smooth=200):
    env = gym.make("Walker2d-v4", render_mode=None)

    # 关键点：直接在 load 时传入 env，并用 custom_objects 兜底旧字段
    model = SAC.load(
        model_path,
        env=env,                    # ✅ 直接传 env，SB3 会自己包 Monitor/DummyVecEnv，且允许并行数不同
        custom_objects={
            "observation_space": env.observation_space,  # 覆盖旧存档里的 space
            "action_space": env.action_space,
            "lr_schedule": (lambda _: 3e-4),             # 兜底学习率调度器
            "_last_obs": None,
            "_last_episode_starts": None,
            "_last_original_obs": None,
        },
        device="cpu",
    )

    obs, _ = env.reset()
    step_rewards, cum_expensive, cum_timelimit = [], [], []
    e_cnt = t_cnt = 0

    for t in range(1, total_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        step_rewards.append(float(r))

        if terminated and not info.get("TimeLimit.truncated", False):
            e_cnt += 1
        if truncated:
            t_cnt += 1

        cum_expensive.append(e_cnt)
        cum_timelimit.append(t_cnt)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    rew_ma = moving_average(step_rewards, smooth)
    L = len(rew_ma)
    steps = np.arange(1, L + 1)
    return steps, rew_ma, np.asarray(cum_expensive[:L], float), np.asarray(cum_timelimit[:L], float)


def main():
    ap = argparse.ArgumentParser(description="Plot reward (MA) & cumulative resets vs steps")
    ap.add_argument("--models", nargs="+", required=True,
                    help="模型路径列表，如: ./models/walker2d_sac_exp1.zip ./models/walker2d_sac_exp2.zip ...")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="每个模型的图例标签，数量与 --models 一致")
    ap.add_argument("--steps", type=int, default=200000,
                    help="每个模型最多运行多少步（steps）")
    ap.add_argument("--smooth", type=int, default=200,
                    help="reward 移动平均窗口（step）")
    ap.add_argument("--out_png", type=str, default="./reward_resets_vs_steps.png",
                    help="输出图片文件")
    args = ap.parse_args()

    assert len(args.models) == len(args.labels), "models 与 labels 数量必须一致"

    fig, ax1 = plt.subplots(figsize=(9, 4.8))  # 左轴：reward
    ax2 = ax1.twinx()                         # 右轴：resets

    for model_path, label in zip(args.models, args.labels):
        steps, rew_ma, cum_exp, cum_tl = run_trajectory(
            model_path=model_path, total_steps=args.steps, smooth=args.smooth
        )
        # 实线：reward
        ax1.plot(steps, rew_ma, label=f"{label}: reward", linewidth=1.5)
        # 虚线：昂贵重置；点划线：TimeLimit（可按需注释掉其一）
        ax2.plot(steps, cum_exp, linestyle="--", label=f"{label}: exp. resets", linewidth=1.2)
        ax2.plot(steps, cum_tl, linestyle=":", label=f"{label}: timelimit", linewidth=1.2)

    ax1.set_xlabel("steps")
    ax1.set_ylabel("reward (moving average)")
    ax2.set_ylabel("cumulative resets")

    # 合并图例：把两个坐标轴的句柄合在一起
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, ncol=2)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200)
    print(f"[Saved] {args.out_png}")

if __name__ == "__main__":
    main()
