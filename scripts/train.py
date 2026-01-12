import gymnasium as gym
import argparse
import os
import time
import numpy as np

# stable-baselines3 (SB3) 的标准组件
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# 导入我们自定义的回调和包装器
from src.common import ResetLoggerCallback
from src.cost_wrapper import CostPunishWrapper

# --- 实验三所需的 SB3 内部组件 ---
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import safe_mean, get_latest_run_id

# --- 实验三的“决策”常量 ---

# 1. 决策点一：预防性触发区
# 官方失败区: height < 0.8 或 > 2.0, angle > 1.0
# 我们的预防触发区 (在“失败”之前触发)
TRIGGER_LOW_HEIGHT = 0.84  # (官方失败是 0.8)
TRIGGER_HIGH_HEIGHT = 1.92 # (官方失败是 2.0)
TRIGGER_ANGLE = 0.96       # (官方失败是 1.0)
# TRIGGER_LOW_HEIGHT = 0.82  # (官方失败是 0.8)
# TRIGGER_HIGH_HEIGHT = 1.98 # (官方失败是 2.0)
# TRIGGER_ANGLE = 0.98       # (官方失败是 1.0)

# 2. 决策点二：恢复任务
# 恢复成功的阈值（与“家”的距离）
RECOVERY_SUCCESS_THRESHOLD = 0.5 

# 3. 决策点三：恢复限制
MAX_RECOVERY_STEPS = 1000 # 恢复任务的超时步数

# --- 实验三的算法超参数 ---
# (这些通常在 model.learn() 中自动设置)
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 8192
LEARNING_STARTS = 10_000 # 在收集这么多样本之前不开始训练
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4 # 0.0003
TRAIN_FREQ = 1 # 每 1 步训练 1 次

# --- 实验三的辅助函数 ---

def is_in_trigger_zone(obs: np.ndarray) -> bool:
    """
    (决策点一) 检查当前状态是否在我们的“预防触发区”
    Walker2d-v4 的观测量:
    - obs[0] 是 z-coordinate (height)
    - obs[1] 是 angle (torso inclination)
    """
    height = obs[0]
    angle = obs[1]
    
    is_triggered = (
        height < TRIGGER_LOW_HEIGHT
        or height > TRIGGER_HIGH_HEIGHT
        or abs(angle) > TRIGGER_ANGLE
    )
    return is_triggered

def get_recovery_reward(obs: np.ndarray, s_target: np.ndarray) -> float:
    """
    (决策点二) 计算恢复策略的奖励
    奖励 = - (与“家”的距离)
    """
    distance = np.linalg.norm(obs - s_target)
    return -distance


# --- 实验三的训练函数 (已修复第三次) ---

def train_experiment_3(total_timesteps, log_dir, save_path):
    """
    实验三：自主恢复策略的完整训练循环
    """
    print("--- 启动实验三 (自主恢复策略) ---")
    
    # 1. 初始化环境 (必须使用 n_envs=1)
    env = make_vec_env("Walker2d-v4", n_envs=1, seed=0)

    # 2. 获取“家”的状态 (决策点二)
    s_target = env.reset()[0] 
    print(f"“家” (s_target) 的维度: {s_target.shape}")

    # 3. 初始化两个策略和两个 Buffer
    policy_F = SAC(
        "MlpPolicy", 
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_starts=LEARNING_STARTS,
        verbose=0
    )
    
    policy_R = SAC(
        "MlpPolicy", 
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_starts=LEARNING_STARTS,
        verbose=0
    )
    
    buffer_F = policy_F.replay_buffer
    buffer_R = policy_R.replay_buffer

    # 4. 初始化日志和统计
    # (根据你的要求，自动创建 SAC_1, SAC_2 等子文件夹)
    run_id = get_latest_run_id(log_dir, "SAC")
    run_log_dir = os.path.join(log_dir, f"SAC_4")
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Logging to {run_log_dir}") 

    new_logger = configure(run_log_dir, ["stdout", "tensorboard"])
    policy_F.set_logger(new_logger)
    policy_R.set_logger(new_logger)
    
    stats_expensive_resets = 0
    stats_artificial_resets = 0
    stats_recovery_successes = 0
    stats_recovery_triggers_safe = 0
    stats_recovery_triggers_time = 0
    
    # 5. 主训练循环
    print(f"开始训练 {total_timesteps} 步...")
    
    obs = env.reset()
    current_mode = "FORWARD"
    current_recovery_steps = 0
    
    ep_reward_F_list = []
    ep_reward_F = 0.0
    
    start_time = time.time()

    for step in range(1, int(total_timesteps) + 1):
        
        if current_mode == "FORWARD":
            # (A) Policy_F 正在运行
            
            action, _ = policy_F.predict(obs, deterministic=False)
            new_obs, reward, dones, infos = env.step(action)
            
            terminated_signal = dones[0] and not infos[0].get("TimeLimit.truncated", False)
            buffer_F.add(obs[0], new_obs[0], action[0], reward[0], terminated_signal, infos)
            
            ep_reward_F += reward[0]
            
            is_fail = dones[0] and not infos[0].get("TimeLimit.truncated", False)
            is_time_limit = infos[0].get("TimeLimit.truncated", False)
            
            is_heuristic_trigger = is_in_trigger_zone(new_obs[0])
            
            # 4. 状态转移
            if is_fail:
                stats_expensive_resets += 1
                obs = env.reset()
                ep_reward_F_list.append(ep_reward_F)
                ep_reward_F = 0.0
                
            elif is_heuristic_trigger:
                # --- 这是我们的修复点 ---
                # 当安全触发器激活时，
                # `Policy_F` 的这个回合也算“结束”了。
                # 我们必须记录它的得分并重置累加器。
                
                stats_recovery_triggers_safe += 1
                current_mode = "RECOVERY"
                current_recovery_steps = 0
                obs = new_obs
                
                # --- 修复代码开始 ---
                ep_reward_F_list.append(ep_reward_F)
                ep_reward_F = 0.0
                # --- 修复代码结束 ---
                
            elif is_time_limit:
                stats_recovery_triggers_time += 1
                current_mode = "RECOVERY"
                current_recovery_steps = 0
                obs = new_obs
                ep_reward_F_list.append(ep_reward_F)
                ep_reward_F = 0.0
                
            else:
                obs = new_obs

        elif current_mode == "RECOVERY":
            # (B) Policy_R 正在运行
            
            action, _ = policy_R.predict(obs, deterministic=False)
            new_obs, _, dones, infos = env.step(action)
            
            current_recovery_steps += 1
            
            reward_R = get_recovery_reward(new_obs[0], s_target)
            terminated_signal = dones[0] and not infos[0].get("TimeLimit.truncated", False)
            buffer_R.add(obs[0], new_obs[0], action[0], reward_R, terminated_signal, infos)
            
            is_fail = dones[0] and not infos[0].get("TimeLimit.truncated", False)
            is_timeout = current_recovery_steps >= MAX_RECOVERY_STEPS
            
            distance_to_home = np.linalg.norm(new_obs[0] - s_target)
            is_success = distance_to_home < RECOVERY_SUCCESS_THRESHOLD
            
            # 4. 状态转移
            if is_fail:
                stats_expensive_resets += 1
                obs = env.reset()
                current_mode = "FORWARD"
                
            elif is_timeout:
                stats_artificial_resets += 1
                obs = env.reset()
                current_mode = "FORWARD"
                
            elif is_success:
                stats_recovery_successes += 1
                obs = env.reset()
                current_mode = "FORWARD"
                
            else:
                obs = new_obs

        # --- 训练 ---
        if step >= LEARNING_STARTS:
            if step % TRAIN_FREQ == 0:
                policy_F.train(gradient_steps=1, batch_size=BATCH_SIZE)
                policy_R.train(gradient_steps=1, batch_size=BATCH_SIZE)
        
        # --- 日志记录 ---
        if step % 5000 == 0:
            fps = int(step / (time.time() - start_time))
            avg_f_reward = safe_mean(ep_reward_F_list) if ep_reward_F_list else 0.0
            ep_reward_F_list = []
            
            print(f"--- 总步数: {step}/{int(total_timesteps)} ---")
            print(f"FPS: {fps}")
            print(f"当前状态: {current_mode}")
            print(f"Policy_F 平均回报 (过去 5k 步): {avg_f_reward:.2f}")
            print(f"昂贵重置 (累计): {stats_expensive_resets}")
            print(f"人工重置 (累计): {stats_artificial_resets}")
            print(f"恢复成功 (累计): {stats_recovery_successes}")
            print(f"触发-安全 (累计): {stats_recovery_triggers_safe}")
            print(f"触发-时间 (累计): {stats_recovery_triggers_time}")
            
            new_logger.record("train/fps", fps)
            new_logger.record("train/policy_F_reward", avg_f_reward)
            new_logger.record("train_stats/cumulative_expensive_resets", stats_expensive_resets)
            new_logger.record("train_stats/cumulative_artificial_resets", stats_artificial_resets)
            new_logger.record("train_stats/cumulative_recovery_success", stats_recovery_successes)
            new_logger.record("train_stats/triggers_safe", stats_recovery_triggers_safe)
            new_logger.record("train_stats/triggers_time", stats_recovery_triggers_time)
            new_logger.dump(step)

    # 6. 训练结束
    env.close()
    
    policy_F.save(save_path)
    
    print("--- 实验三结束 ---")
    print(f"模型 (Policy_F) 已保存到: {save_path}")
    print(f"昂贵重置总次数: {stats_expensive_resets}")
    print(f"人工重置总次数: {stats_artificial_resets}")
    print(f"恢复成功总次数: {stats_recovery_successes}")


# --- 实验一和二的实现 (与之前相同) ---

def train_experiment_1_or_2(experiment_num, total_timesteps, log_dir, save_path):
    """
    训练实验一或实验二
    """
    print(f"--- 启动实验 {experiment_num} ---")
    
    # 1. 创建日志目录
    tensorboard_log_dir = os.path.join(log_dir, f"exp{experiment_num}")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # 2. 创建回调函数 (用于统计重置)
    reset_logger = ResetLoggerCallback()

    # 3. 创建环境
    if experiment_num == 1:
        # 实验一：免费基线
        env = make_vec_env("Walker2d-v4", n_envs=24, seed=0)
    
    elif experiment_num == 2:
        # 实验二：成本惩罚
        env = make_vec_env(
            "Walker2d-v4", 
            n_envs=24, 
            seed=0,
            wrapper_class=CostPunishWrapper,
            wrapper_kwargs=dict(cost=-1000.0) # 传入高额惩罚
        )
    
    # 4. 创建模型 (SAC)
    model = SAC(
        "MlpPolicy", 
        env,
        verbose=1, 
        tensorboard_log=tensorboard_log_dir,
        learning_starts=LEARNING_STARTS,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
    )

    # 5. 开始训练
    print(f"开始训练 {total_timesteps} 步...")
    print(f"TensorBoard 日志将保存在: {tensorboard_log_dir}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=reset_logger, # 挂载我们的重置统计器
        log_interval=10
    )

    # 6. 保存模型
    final_save_path = save_path # 使用完整的路径
    model.save(final_save_path)
    print(f"--- 实验 {experiment_num} 结束 ---")
    print(f"模型已保存到: {final_save_path}")
    print(f"昂贵重置总次数: {reset_logger.expensive_resets_count}")
    print(f"人工重置总次数: {reset_logger.artificial_resets_count}")


# --- 主函数 (更新以正确处理路径) ---

def main():
    parser = argparse.ArgumentParser(description="运行 Walker2d 强化学习实验")
    parser.add_argument(
        "--experiment", 
        type=int, 
        required=True, 
        choices=[1, 2, 3],
        help="要运行的实验编号 (1, 2, 或 3)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1_000_000,
        help="总训练步数"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="outputs/logs",
        help="TensorBoard 日志的保存目录"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="outputs/models/walker2d_sac",
        help="模型保存的基本路径 (会自动添加 _expN.zip)"
    )
    
    args = parser.parse_args()
    
    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # 为每个实验构造唯一的日志和保存路径
    exp_log_dir = os.path.join(args.log_dir, f"exp{args.experiment}")
    exp_save_path = f"{args.save_path}_exp{args.experiment}.zip"
    
    os.makedirs(exp_log_dir, exist_ok=True)

    if args.experiment == 1 or args.experiment == 2:
        # 实验 1 和 2 的日志由 SB3 在其函数内部处理
        # (我们把 `tensorboard_log` 参数传给了 SAC)
        # 所以我们把 *基础* 日志目录传给它
        train_experiment_1_or_2(args.experiment, args.steps, args.log_dir, exp_save_path)
    
    elif args.experiment == 3:
        # 实验 3 使用我们自定义的 logger，
        # 它需要一个 *精确* 的日志目录
        train_experiment_3(args.steps, exp_log_dir, exp_save_path)

if __name__ == "__main__":
    main()