import torch
import torch.nn as nn
import torch.nn.functional as F


# 双层CNN
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, mode='both'):
        super(MultimodalClassifier, self).__init__()
        self.mode = mode

        # 定义音频分支
        self.audio_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # 定义图像分支
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # 获取输出尺寸以确定全连接层的输入
        audio_out_dim = 64 * 8 * 8  # 修改此处以适应不同的输入大小
        image_out_dim = 64 * 8 * 8  # 修改此处以适应不同的输入大小

        # 全连接层
        self.fc_audio = nn.Linear(audio_out_dim, num_classes)
        self.fc_image = nn.Linear(image_out_dim, num_classes)
        self.fc_fusion = nn.Sequential(
            nn.Linear(audio_out_dim + image_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio_data=None, image_data=None, use_images=True, use_ais=True):
        # 如果只使用音频数据
        if self.mode == 'audio' and use_ais and audio_data is not None:
            audio_features = self.audio_branch(audio_data)
            output = self.fc_audio(audio_features)

        # 如果只使用图像数据
        elif self.mode == 'image' and use_images and image_data is not None:
            image_features = self.image_branch(image_data)
            output = self.fc_image(image_features)

        # 如果使用两种模态数据
        elif self.mode == 'both' and use_ais and use_images and audio_data is not None and image_data is not None:
            audio_features = self.audio_branch(audio_data)
            image_features = self.image_branch(image_data)
            fused_features = torch.cat((audio_features, image_features), dim=1)
            output = self.fc_fusion(fused_features)

        else:
            raise ValueError("Invalid mode or missing input data for the selected mode.")

        return output


# 单层CNN
class MultimodalClassifier_Single(nn.Module):
    def __init__(self, num_classes, mode='both'):
        super(MultimodalClassifier_Single, self).__init__()
        self.mode = mode

        # 定义单层卷积的音频
        self.shared_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU()
        )

        # 图像分支：输入为3通道的卷积层
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU()
        )

        # 全连接层
        self.fc_audio = nn.Linear(128, num_classes)
        self.fc_image = nn.Linear(128, num_classes)
        self.fc_fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, audio_data=None, image_data=None, use_images=True, use_ais=True):
        # 如果只使用音频数据
        if self.mode == 'audio' and use_ais and audio_data is not None:
            audio_features = self.shared_branch(audio_data)
            output = self.fc_audio(audio_features)

        # 如果只使用图像数据
        elif self.mode == 'image' and use_images and image_data is not None:
            image_features = self.image_branch(image_data)
            output = self.fc_image(image_features)

        # 如果使用两种模态数据
        elif self.mode == 'both' and use_ais and use_images and audio_data is not None and image_data is not None:
            audio_features = self.shared_branch(audio_data)
            image_features = self.image_branch(image_data)
            fused_features = torch.cat((audio_features, image_features), dim=1)
            output = self.fc_fusion(fused_features)

        else:
            raise ValueError("Invalid mode or missing input data for the selected mode.")

        return output