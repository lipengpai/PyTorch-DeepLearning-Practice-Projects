# CatDogClassification 项目

完整基于PyTorch的猫狗图片分类(CNN)，含数据处理、训练推理。  
结构脚本化，支持命令行/脚本方式一键全流程。

## 运行步骤

1. 准备数据集至 ./datasets/PetImages/ 下（结构 Cat/ Dog/）
2. 数据划分：`python dataset_process.py`
3. 训练模型：`python train.py`
4. 推理体验：`python inference.py`

## 目录结构

见本项目目录树。  
核心模型结构详见 model.py。

---

如需更多细节和可视化分析请参考注释和代码。