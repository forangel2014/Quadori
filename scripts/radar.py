import numpy as np
import matplotlib.pyplot as plt

def plot_radar(labels, chatgpt_data, llama_data, quadori_data, name, ax):
    for i in range(len(labels)):
        if labels[i] == '1-self-BLEU-2':
            chatgpt_data[i] = 100 - chatgpt_data[i]
            llama_data[i] = 100 - llama_data[i]
            quadori_data[i] = 100 - quadori_data[i]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    chatgpt_data += chatgpt_data[:1]  # Repeat the first value to close the circle
    llama_data += llama_data[:1]  # Repeat the first value to close the circle
    quadori_data += quadori_data[:1]  # Repeat the first value to close the circle
    angles += angles[:1]  # Repeat the first angle to close the circle
    ax.plot(angles, chatgpt_data, label='ChatGPT')
    ax.plot(angles, llama_data, label='LLama2')
    ax.plot(angles, quadori_data, label='Quadori')
    ax.fill(angles, chatgpt_data, alpha=0.25)
    ax.fill(angles, llama_data, alpha=0.25)
    ax.fill(angles, quadori_data, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_title(name)
    ax.legend(loc='upper right', bbox_to_anchor=(1.08, 1.08))  # 调整图例位置
    return ax

def plot_radar_OpenRI():
    labels = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'ROUGE-L', 'METEOR', '1-self-BLEU-2', 'ES']
    chatgpt_data = [34.03, 16.45, 5.49, 41.44, 33.91, 81.72, 23.69]
    llama_data =   [47.81, 29.04, 1.30, 50.52, 44.41, 83.83, 23.30]
    quadori_data = [50.52, 26.88, 7.32, 53.36, 43.49, 87.36, 29.46]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax = plot_radar(labels, chatgpt_data, llama_data, quadori_data, "OpenRI", ax)
    plt.title("OpenRI")
    plt.show()
    plt.savefig("OpenRI.pdf")

plot_radar_OpenRI()

def plot_radar_RE():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    labels = ['BLEU-2', 'ROUGE-L', 'METEOR', '1-self-BLEU-2']

    # 列1数据
    chatgpt_data = [3.29, 26.33, 16.28, 81.79]
    llama_data = []
    quadori_data = [11.40, 42.68, 28.20, 88.53]
    axes[0] = plot_radar(labels, chatgpt_data, llama_data, quadori_data, "FewRel", axes[0])

    # 列2数据
    chatgpt_data = [2.72, 19.28, 12.74, 79.36]
    llama_data = []
    quadori_data = [10.73, 40.30, 29.26, 88.34]
    axes[1] = plot_radar(labels, chatgpt_data, llama_data, quadori_data, "NYT10", axes[1])

    # 列3数据
    chatgpt_data = [3.31, 21.32, 13.67, 79.73]
    llama_data = []
    quadori_data = [7.84, 37.69, 25.15, 88.07]
    axes[2] = plot_radar(labels, chatgpt_data, llama_data, quadori_data, "WIKI80", axes[2])

    # 列4数据
    chatgpt_data = [3.94, 20.87, 14.23, 79.56]
    llama_data = []
    quadori_data = [9.94, 40.52, 27.80, 88.33]
    axes[3] = plot_radar(labels, chatgpt_data, llama_data, quadori_data, "TREx", axes[3])

    # 列5数据
    chatgpt_data = [3.89, 11.69, 16.06, 78.45]
    llama_data = []
    quadori_data = [7.93, 37.09, 23.28, 87.89]
    axes[4] = plot_radar(labels, chatgpt_data, llama_data, quadori_data, "Google-RE", axes[4])

    # 列6数据
    chatgpt_data = [3.29, 26.33, 16.28, 79.19]
    llama_data = []
    quadori_data = [8.52, 41.50, 29.30, 87.11]
    axes[5] = plot_radar(labels, chatgpt_data, llama_data, quadori_data, "SemEval", axes[5])

    plt.tight_layout()
    #plt.suptitle("RE")
    plt.show()
    plt.savefig("RE.pdf")

plot_radar_RE()