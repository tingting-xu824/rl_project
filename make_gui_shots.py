# --- minimal NumPy2->NumPy1 pickle shim: must come BEFORE importing SB3/torch ---
import sys, importlib
try:
    import numpy._core  # NumPy 2.x: already present -> nothing to do
except Exception:
    import numpy.core as _ncore
    sys.modules['numpy._core'] = _ncore
    for _n in ('numeric','fromnumeric','_methods','multiarray','umath','overrides','shape_base'):
        try:
            sys.modules[f'numpy._core.{_n}'] = importlib.import_module(f'numpy.core.{_n}')
        except Exception:
            pass
# -------------------------------------------------------------------------------

"""
make_gui_shots.py (rgb_array only)
- 只创建离屏渲染环境(render_mode="rgb_array")，避免 human overlay 崩溃
- 可选加载 SAC 模型；不提供则用随机动作
- 在指定步数抓取 PNG 截图
"""

import os
import argparse
import time
from typing import List, Optional

import numpy as np
import gymnasium as gym

# 可选：稳定基线
try:
    from stable_baselines3 import SAC
    _HAVE_SB3 = True
except Exception:
    _HAVE_SB3 = False


def run_and_capture(
    model_path: Optional[str],
    total_steps: int,
    shot_steps: List[int],
    out_files: List[str],
    seed: int = 42,
    sleep_each_step: float = 0.0,
):
    assert len(shot_steps) == len(out_files), "shot_steps 与 out 文件数量需一致"

    # 只用离屏环境，避免 human overlay
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    # 选择动作来源
    model = None
    if model_path:
        if not _HAVE_SB3:
            raise RuntimeError("未安装 stable_baselines3，无法加载模型。请 `pip install stable-baselines3`")
        model = SAC.load(
            model_path,
            env=env,  # 传入当前环境，避免空间反序列化不一致
            custom_objects={
                "lr_schedule": (lambda _: 3e-4),
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "_last_obs": None,
                "_last_episode_starts": None,
                "_last_original_obs": None,
            },
        )

    step = 0
    saved = [False] * len(shot_steps)

    while step < total_steps:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        # 抓图
        for i, s in enumerate(shot_steps):
            if (not saved[i]) and (step >= s):
                frame = env.render()  # HxWx3
                if frame is not None:
                    import imageio
                    imageio.imwrite(out_files[i], frame)
                    print(f"[Saved] step={step} -> {out_files[i]}")
                    saved[i] = True

        if terminated or truncated:
            obs, _ = env.reset()

        if sleep_each_step > 0:
            time.sleep(sleep_each_step)

        if all(saved):
            break

    env.close()


def parse_args():
    p = argparse.ArgumentParser(description="用 rgb_array 抓取 Walker2d 截图（无 GUI）")
    p.add_argument("--model_path", type=str, default=None,
                   help="可选：SAC 模型路径；不传则随机动作")
    p.add_argument("--steps", type=int, default=200, help="最多运行多少步")
    p.add_argument("--shot_steps", type=int, nargs=2, default=[50, 150],
                   help="两张图的抓取步数，例如: --shot_steps 50 150")
    p.add_argument("--out", type=str, nargs=2, default=["shots1.png", "shots2.png"],
                   help="两张图的输出文件名，例如: --out shots1.png shots2.png")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="每步 sleep 秒数（可为 0）")
    return p.parse_args()


def main():
    # 避免强制 EGL
    os.environ.pop("MUJOCO_GL", None)

    args = parse_args()
    print("--- 仅 rgb_array 运行并抓图 ---")
    print(f"model_path = {args.model_path}")
    print(f"steps = {args.steps}, shot_steps = {args.shot_steps}, out = {args.out}")

    run_and_capture(
        model_path=args.model_path,
        total_steps=args.steps,
        shot_steps=list(args.shot_steps),
        out_files=list(args.out),
        seed=args.seed,
        sleep_each_step=args.sleep,
    )

    print("--- 完成 ---")


if __name__ == "__main__":
    main()