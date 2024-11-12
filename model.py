import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalClassifier1(nn.Module):
    def __init__(self, num_classes, mode='both'):
        super(MultimodalClassifier1, self).__init__()
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

        # 使用预训练的 ResNet-18 作为图像分支
        self.image_branch = models.resnet18(pretrained=True)
        self.image_branch.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.image_branch.fc = nn.Identity()  # 移除 ResNet 的分类头，仅提取特征

        # 计算音频分支和图像分支输出维度
        audio_out_dim = 64 * 8 * 8  # 根据实际情况调整
        image_out_dim = 512  # ResNet-18 的特征输出为 512 维

        # 定义全连接层
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
