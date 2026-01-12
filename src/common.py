import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ResetLoggerCallback(BaseCallback):
    """
    一个自定义的回调函数，用于统计“昂贵重置”和“人工重置”的次数。
    
    它在训练过程中会周期性地在TensorBoard或控制台中打印这些统计数据。
    """
    def __init__(self, verbose=0):
        super(ResetLoggerCallback, self).__init__(verbose)
        self.expensive_resets_count = 0
        self.artificial_resets_count = 0

    def _on_step(self) -> bool:
        """
        这个函数会在 model.learn() 的每一步被调用。
        """
        # `dones` 是一个数组，对应所有并行的环境。
        # 我们检查是否有任何环境在这一步 "done" (terminated 或 truncated)
        for i, done in enumerate(self.locals["dones"]):
            if done:
                # 如果 "done" 为 True，我们需要区分是哪种 "done"
                
                # SB3 会把环境信息放在 `infos` 列表里
                info = self.locals["infos"][i]
                
                # `TimeLimit.truncated` 是 Gym 标准，
                # 当达到1000步时，它会是 True。
                is_truncated = info.get("TimeLimit.truncated", False)

                if is_truncated:
                    # 这是1000步的人工重置
                    self.artificial_resets_count += 1
                else:
                    # 这不是因为时间耗尽，而是因为摔倒了
                    # 这就是“昂贵重置” (Dilemma Reset)
                    self.expensive_resets_count += 1
        
        # 记录到日志中 (例如 TensorBoard)
        # self.num_timesteps 是当前的总步数
        self.logger.record("train_stats/cumulative_expensive_resets", self.expensive_resets_count)
        self.logger.record("train_stats/cumulative_artificial_resets", self.artificial_resets_count)
        self.logger.record("train_stats/total_hard_resets", self.expensive_resets_count + self.artificial_resets_count)

        return True # 继续训练