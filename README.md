# YOLOv8 鸡蛋状态分类

## 概述
本项目使用 YOLOv8 目标检测模型对鸡蛋状态进行分类，主要类别包括：
- **正常鸡蛋**
- **死胎鸡蛋**
- **无精鸡蛋**

模型基于自定义数据集进行训练，并应用 **微调技术** 以优化性能。

## 功能特点
- **数据集拆分**：自动将图像划分为训练集、验证集和测试集。
- **模型训练**：利用超参数微调 YOLOv8 模型。
- **模型评估**：使用标准目标检测指标评估模型性能。
- **实时推理**：在视频流上进行实时检测。

## 安装
### 先决条件
请确保您的 Python 环境已安装以下依赖项：
```bash
pip install ultralytics opencv-python torch torchvision numpy scikit-learn
```

## 数据集准备
1. 按以下结构组织图像和标签：
   ```
   dataset/
   ├── images/
   │   ├── egg1.jpg
   │   ├── egg2.jpg
   │   └── ...
   ├── labels/
   │   ├── egg1.txt
   │   ├── egg2.txt
   │   └── ...
   ```
2. 运行数据集拆分函数：
   ```python
   split_dataset('dataset/images', 'dataset/labels', 'dataset_split')
   ```

## 训练模型
使用微调技术训练 YOLOv8 模型：
```python
train_yolov8(model_type='yolov8s', data_yaml='data.yaml', epochs=50, batch_size=8, lr=0.001, weight_decay=0.0005)
```

## 评估模型
评估训练好的模型：
```python
evaluate_model(model_path='runs/train/exp/weights/best.pt', data_yaml='data.yaml')
```

## 实时推理
使用摄像头或视频文件进行实时检测：
```python
real_time_inference(model_path='runs/train/exp/weights/best.pt', video_source=0)
```

## 使用的微调技术
- **学习率调整**：优化收敛稳定性。
- **权重衰减**：防止过拟合。
- **优化器选择**：使用 SGD 以增强泛化能力。

## 未来改进方向
- 实现云端推理。
- 集成额外的鸡蛋质量检测指标。

## 贡献者
Yubo Liu
