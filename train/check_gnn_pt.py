import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# 親ディレクトリをモジュール検索パスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH

state_dict = torch.load(MODEL_PATH)
param_names = list(state_dict.keys())

num_params = len(param_names)
cols = 3  # 表示列数（調整可）
rows = (num_params + cols - 1) // cols

plt.figure(figsize=(cols * 6, rows * 4))
plt.suptitle("Parameter Distributions (Histogram)", fontsize=18)

for i, key in enumerate(param_names):
    tensor = state_dict[key].cpu().view(-1).numpy()

    ax = plt.subplot(rows, cols, i + 1)
    ax.hist(tensor, bins=50, color="skyblue", edgecolor="black")
    ax.set_title(f"{key}\nmean={np.mean(tensor):.2e}, std={np.std(tensor):.2e}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
