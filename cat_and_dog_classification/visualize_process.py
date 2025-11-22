import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from model import Net  # 确保你的目录下有 model.py

# --- 配置 ---
MODEL_PATH = './models/model_weights.pth'
IMAGE_PATH = './data/dog.jpg'  # 替换成你想分析的图片路径 (建议选一张特征明显的图)
SAVE_NAME = 'process_visualization.png'

def visualize_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. 准备图片预处理
    # 注意：这里只做 Resize 和 Tensor化，不做随机增强，保证可视化的一致性
    preprocess = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    raw_img = Image.open(IMAGE_PATH).convert('RGB')
    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

    # 3. 获取中间层特征 (Hook 机制)
    # 我们想看第一层卷积(conv1)和第三层卷积(conv3)输出了什么
    feature_maps = {}
    def get_activation(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()
        return hook

    # 注册钩子
    model.conv1[0].register_forward_hook(get_activation('conv1')) # 浅层特征
    model.conv3[0].register_forward_hook(get_activation('conv3')) # 深层特征

    # 4. 前向传播
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    # 5. 绘图
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3) # 2行3列布局

    # --- A. 左图：原始图片 ---
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(raw_img)
    ax_orig.set_title("Input Image", fontsize=14)
    ax_orig.axis('off')

    # --- B. 中图：特征图可视化 ---
    # 辅助函数：将多通道特征图显示为网格
    def plot_filters(ax, layer_name, feats, num_filters=4):
        feats = feats.cpu().squeeze(0) # [C, H, W]
        # 取前4个通道可视化
        display_grid = torch.cat([feats[i] for i in range(num_filters)], dim=1)
        ax.imshow(display_grid, cmap='viridis')
        ax.set_title(f"Feature Maps: {layer_name}\n(What the AI 'Sees')", fontsize=12)
        ax.axis('off')

    ax_conv1 = fig.add_subplot(gs[0, 1])
    plot_filters(ax_conv1, 'Conv1 (Edges)', feature_maps['conv1'])

    ax_conv3 = fig.add_subplot(gs[1, 1])
    plot_filters(ax_conv3, 'Conv3 (Shapes)', feature_maps['conv3'])

    # --- C. 右图：预测概率 ---
    ax_prob = fig.add_subplot(gs[:, 2])
    classes = ['Cat', 'Dog']
    probs = probabilities.cpu().numpy()
    colors = ['#FF9999', '#66B2FF']
    
    bars = ax_prob.bar(classes, probs, color=colors, alpha=0.8, width=0.5)
    ax_prob.set_ylim(0, 1.1)
    ax_prob.set_title("Model Confidence", fontsize=14)
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['right'].set_visible(False)
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        ax_prob.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1%}', ha='center', va='bottom', fontsize=12)

    # 保存
    plt.suptitle(f"Prediction Analysis: {os.path.basename(IMAGE_PATH)}", fontsize=16)
    plt.savefig(SAVE_NAME, dpi=150)
    print(f"可视化分析图已保存至: {SAVE_NAME}")
    plt.show()

if __name__ == "__main__":
    visualize_inference()
