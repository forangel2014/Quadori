import matplotlib.pyplot as plt

data = [
    ["Quadori w/o ETP", 47.35, 22.58, 5.29, 51.71, 40.84, 88.07, 30.67],
    ["Quadori w/o SSTS", 48.56, 25.03, 6.27, 52.09, 42.14, 88.51, 27.38],
    ["Quadori w/o DPP", 49.62, 25.97, 7.15, 52.96, 43.15, 89.13, 28.36],
    ["Quadori", 50.52, 26.88, 7.32, 53.36, 43.49, 87.36, 29.46]
]

labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'ROUGE-L', 'METEOR', 'self-BLEU-2', 'ES']

# 提取折线名称和数据
line_names = [row[0] for row in data]
line_data = [row[1:] for row in data]

# 绘制折线图
for i, line in enumerate(line_data):
    plt.plot(labels, line, label=line_names[i])

# 添加图例和标签
plt.legend()
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Comparison')

# 显示图形或保存为图像
plt.show()
plt.savefig("ablation.png")