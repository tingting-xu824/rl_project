import gymnasium as gym

class CostPunishWrapper(gym.Wrapper):
    """
    一个环境包装器，用于实现实验二：
    当发生“昂贵重置”(摔倒)时，施加一个巨大的负奖励。
    """
    def __init__(self, env: gym.Env, cost: float):
        super().__init__(env)
        self.cost = cost

    def step(self, action):
        """
        在环境的 step 函数上加一层逻辑
        """
        # 运行原始环境的 step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 检查是否是“昂贵重置”
        # terminated=True (摔倒了)
        # 并且 *不是* 因为 TimeLimit (truncated=False)
        if terminated and not info.get("TimeLimit.truncated", False):
            # 这是昂贵重置！施加惩罚。
            reward += self.cost
            
        return obs, reward, terminated, truncated, info