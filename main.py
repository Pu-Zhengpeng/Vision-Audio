import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataset import MultimodalDataset  # 导入自定义的数据集类
from config import *
import matplotlib.pyplot as plt

from model0 import MultimodalClassifier,MultimodalClassifier_Single
from model import MultimodalClassifier1
from model2 import MultimodalClassifierTransformer

# 设置随机种子，确保每次划分结果相同
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# 数据预处理变换
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# 训练函数
def train(model, train_loader, optimizer, criterion, device, mode='both'):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    for batch_idx, (audio_data, image_data, labels) in enumerate(train_loader):
        # 将数据移到GPU（如果有）
        audio_data, image_data, labels = audio_data.to(device), image_data.to(device), labels.to(device)

        # 清空优化器的梯度
        optimizer.zero_grad()

        # 根据选择的模式，决定输入数据的使用方式
        if mode == 'audio':
            output = model(audio_data=audio_data, use_images=False, use_ais=True)
        elif mode == 'image':
            output = model(image_data=image_data, use_images=True, use_ais=False)
        else:  # 默认使用多模态
            output = model(audio_data=audio_data, image_data=image_data, use_images=True, use_ais=True)

        # 计算损失
        loss = criterion(output, labels)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        # 统计损失
        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    print(f'Training Loss: {average_loss:.4f}')
    return average_loss

# 测试函数
def test(model, test_loader, criterion, device, mode='both'):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 在评估时不需要计算梯度
        for batch_idx, (audio_data, image_data, labels) in enumerate(test_loader):
            # 将数据移到GPU（如果有）
            audio_data, image_data, labels = audio_data.to(device), image_data.to(device), labels.to(device)

            # 根据选择的模式，决定输入数据的使用方式
            if mode == 'audio':
                output = model(audio_data=audio_data, use_images=False, use_ais=True)
            elif mode == 'image':
                output = model(image_data=image_data, use_images=True, use_ais=False)
            else:  # 默认使用多模态
                output = model(audio_data=audio_data, image_data=image_data, use_images=True, use_ais=True)

            # 计算损失
            loss = criterion(output, labels)

            # 统计损失
            running_loss += loss.item()

            # 获取预测结果
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return average_loss, accuracy


if __name__ == '__main__':
    # 加载数据集
    dataset = MultimodalDataset(AUDIO_DATA_PATH, IMAGE_DATA_PATH, transform=data_transforms)

    # 划分训练集和测试集，比例为7:3
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=seed, stratify=dataset.labels)

    # 创建DataLoader实例
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

    # 查看一组train_loader的数据
    audio_data, image_data, labels = next(iter(train_loader))

    # # 打印数据信息
    # print("Audio Data Shape:", audio_data.shape)
    # print("Image Data Shape:", image_data.shape)
    # print("Labels Shape:", labels)
    #
    # # 可视化第一个图像样本
    # first_image = image_data[0].permute(1, 2, 0)  # 将通道维度移到最后
    # plt.imshow(first_image.numpy())
    # plt.title(f"Label: {labels[0]}")
    # plt.show()

    # 实例化模型，优化器，损失函数
    model = MultimodalClassifier_Single(num_classes=num_classes, mode='both').to(DEVICE)   ### 1.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练和测试
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')

        # 训练
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE, mode='both')   ### 2.

        # 测试
        test_loss, test_accuracy = test(model, test_loader, criterion, DEVICE, mode='both')  ### 3.

        # 可选：保存模型
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')