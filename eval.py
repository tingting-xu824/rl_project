# --- minimal NumPy2->NumPy1 pickle shim: must come BEFORE importing SB3/torch ---
import sys, importlib
try:
    import numpy._core  # already NumPy 2.x -> nothing to do
except Exception:
    import numpy.core as _ncore
    sys.modules['numpy._core'] = _ncore
    for _n in ('numeric','fromnumeric','_methods','multiarray','umath','overrides','shape_base'):
        try:
            sys.modules[f'numpy._core.{_n}'] = importlib.import_module(f'numpy.core.{_n}')
        except Exception:
            pass
# -------------------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import argparse
import time

def evaluate_model(model_path, n_episodes=20):
    """
    加载并评估一个已训练的模型。
    """
    print(f"--- 开始评估 ---")
    print(f"加载模型: {model_path}")
    print(f"运行 {n_episodes} 个评估回合...")

    # 1. 加载模型
    # 注意：评估时我们总是使用 *原始* 环境，
    # 即使训练时用了 Wrapper (如实验二)，
    # 我们也想看它在原始环境中的表现。
    env = gym.make("Walker2d-v4", render_mode=None) # 使用 'human' 模式在 Mac 上显示
    model = SAC.load(
        model_path, 
        env=env,
        custom_objects={
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": (lambda _: 3e-4),
        "_last_obs": None,
        "_last_episode_starts": None,
        "_last_original_obs": None,
        },
    )
    
    total_rewards = []
    total_expensive_resets = 0
    total_artificial_resets = 0

    for i in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        
        print(f"开始第 {i+1}/{n_episodes} 回合...")
        
        while not (terminated or truncated):
            # 使用确定性策略进行评估
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            # 渲染 (在你的 Mac 上会弹出一个窗口)
            # env.render() 
            # time.sleep(0.01) # 减慢速度以便观察

            if terminated and not info.get("TimeLimit.truncated", False):
                # 评估过程中也统计昂贵重置
                total_expensive_resets += 1
            
            if truncated:
                total_artificial_resets += 1

        print(f"回合 {i+1} 结束. 回报: {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    env.close()
    
    # 4. 打印最终总结
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_expensive_resets = total_expensive_resets / n_episodes
    avg_artificial_resets = total_artificial_resets / n_episodes
    
    print("\n--- 评估总结 ---")
    print(f"模型: {model_path}")
    print(f"总回合数: {n_episodes}")
    print(f"平均回报: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"平均昂贵重置(摔倒)次数/回合: {avg_expensive_resets:.2f}")
    print(f"平均人工重置(1000步)次数/回合: {avg_artificial_resets:.2f}")
    print("--- 评估结束 ---")


def main():
    parser = argparse.ArgumentParser(description="评估一个已训练的 Walker2d 模型")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="已训练模型的文件路径 (例如: ./models/walker2d_sac_exp1.zip)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=20,
        help="要运行的评估回合数"
    )
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.episodes)

if __name__ == "__main__":
    main()