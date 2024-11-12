# config.py
import os

import torch

# 数据集路径
AUDIO_DATA_PATH = os.path.join("data", "audio_dataset")
IMAGE_DATA_PATH = os.path.join("data", "cifar10_dataset")

# 模型和训练参数
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 10