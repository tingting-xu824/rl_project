# plot_bars.py
# 生成 3 张柱状图：Avg Return、Expensive Resets、TimeLimit(人工重置)

import os
import matplotlib.pyplot as plt
import numpy as np

# === 你的评估结果（可改）===
labels = ["exp1", "exp2", "exp3"]

avg_return = np.array([1338.67, 1278.13, 4231.49])
std_return = np.array([ 936.95,  344.38,  304.90])

expensive_resets = np.array([0.88, 0.15, 0.05])   # 摔倒/回合
timelimit_resets = np.array([0.12, 0.85, 0.95])   # 1000步被截断/回合

out_dir = "./logs"
os.makedirs(out_dir, exist_ok=True)

# 1) 平均回报（带标准差）
plt.figure(figsize=(6,4))
x = np.arange(len(labels))
plt.bar(x, avg_return, yerr=std_return, capsize=6)
plt.xticks(x, labels)
plt.ylabel("Average Return")
plt.title("Avg Return (± Std)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "bar_avg_return.png"), dpi=200)
plt.close()

# 2) 昂贵重置/回合（摔倒）
plt.figure(figsize=(6,4))
plt.bar(x, expensive_resets)
plt.xticks(x, labels)
plt.ylabel("Expensive Resets per Episode")
plt.title("Expensive Resets (Falls)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "bar_expensive_resets.png"), dpi=200)
plt.close()

# 3) 人工重置/回合（TimeLimit 截断）
plt.figure(figsize=(6,4))
plt.bar(x, timelimit_resets)
plt.xticks(x, labels)
plt.ylabel("TimeLimit per Episode")
plt.title("Artificial Resets (TimeLimit)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "bar_timelimit_resets.png"), dpi=200)
plt.close()

print("Done. Saved to ./logs/:")
print(" - bar_avg_return.png")
print(" - bar_expensive_resets.png")
print(" - bar_timelimit_resets.png")
