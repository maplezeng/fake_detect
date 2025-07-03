import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'gossip_scores.txt'

scores = {
    'T': [],
    'IS': [],
    'IP': [],
    'CC': []
}

thresholds = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        m = re.match(r'(\w+): ([0-9.eE+-]+) ([0-9.eE+-]+)', line)
        if m:
            key = m.group(1)
            thr = float(m.group(2))
            val = float(m.group(3))
            if key in scores:
                scores[key].append(val)
                if len(thresholds) < len(scores[key]):
                    thresholds.append(thr)

# 找最短长度
min_len = min(len(v) for v in scores.values())

# 截断所有数组到最短长度
for k in scores:
    scores[k] = scores[k][:min_len]
thresholds = thresholds[:min_len]

# 构造DataFrame
df = pd.DataFrame(scores)
df['Threshold'] = thresholds

# 计算相关系数矩阵
corr = df[['T', 'IS', 'IP', 'CC']].corr()

# 画热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Scores')
plt.show()
