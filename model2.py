import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class MultimodalClassifierTransformer(nn.Module):
    def __init__(self, num_classes, mode='both', audio_dim=32, img_dim=32, nhead=4, num_layers=1):
        super(MultimodalClassifierTransformer, self).__init__()
        self.mode = mode

        # 音频分支的卷积层和 Transformer 编码器
        self.audio_cnn_layer = nn.Conv2d(in_channels=1, out_channels=audio_dim, kernel_size=3, stride=1, padding=1)
        audio_encoder_layer = TransformerEncoderLayer(d_model=audio_dim, nhead=nhead)
        self.audio_branch = TransformerEncoder(audio_encoder_layer, num_layers=num_layers)

        # 图像分支的卷积层和 Transformer 编码器
        self.image_cnn_layer = nn.Conv2d(in_channels=3, out_channels=img_dim, kernel_size=3, stride=1, padding=1)
        image_encoder_layer = TransformerEncoderLayer(d_model=img_dim, nhead=nhead)
        self.image_branch = TransformerEncoder(image_encoder_layer, num_layers=num_layers)

        # 全连接层用于分类
        self.fc_audio = nn.Linear(audio_dim, num_classes)
        self.fc_image = nn.Linear(img_dim, num_classes)
        self.fc_fusion = nn.Sequential(
            nn.Linear(audio_dim + img_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def create_positional_encoding(self, length, dim, batch_size):
        # 生成正确形状的PE: (batch_size, length, dim)
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)  # 扩展为 (batch_size, length, dim)
        return pe

    def forward(self, audio_data=None, image_data=None, use_images=True, use_ais=True):
        # 音频分支
        if audio_data is not None:
            batch_size = audio_data.size(0)
            audio_features = self.audio_cnn_layer(audio_data)  # (batch, audio_dim, 32, 32)
            audio_features = audio_features.flatten(2).permute(2, 0, 1)  # (seq_len, batch, feature_dim)
            audio_seq_len = audio_features.size(0)
            audio_pos_encoding = self.create_positional_encoding(audio_seq_len, audio_features.size(2), batch_size).to(audio_data.device)
            audio_features = audio_features.permute(1, 0, 2) + audio_pos_encoding  # 转换为 (batch, seq_len, feature_dim)
            audio_features = audio_features.permute(1, 0, 2)  # 恢复为 (seq_len, batch, feature_dim) 输入Transformer
            audio_features = self.audio_branch(audio_features)  # Transformer 编码
            audio_features = audio_features.mean(dim=0)  # 池化以获得整体特征

        # 图像分支
        if image_data is not None:
            batch_size = image_data.size(0)
            image_features = self.image_cnn_layer(image_data)  # (batch, img_dim, 32, 32)
            image_features = image_features.flatten(2).permute(2, 0, 1)  # (seq_len, batch, feature_dim)
            image_seq_len = image_features.size(0)
            image_pos_encoding = self.create_positional_encoding(image_seq_len, image_features.size(2), batch_size).to(image_data.device)
            image_features = image_features.permute(1, 0, 2) + image_pos_encoding  # 转换为 (batch, seq_len, feature_dim)
            image_features = image_features.permute(1, 0, 2)  # 恢复为 (seq_len, batch, feature_dim)
            image_features = self.image_branch(image_features)  # Transformer 编码
            image_features = image_features.mean(dim=0)  # 池化以获得整体特征

        # 分类输出
        if self.mode == 'audio' and use_ais and audio_data is not None:
            output = self.fc_audio(audio_features)
        elif self.mode == 'image' and use_images and image_data is not None:
            output = self.fc_image(image_features)
        elif self.mode == 'both' and use_ais and use_images and audio_data is not None and image_data is not None:
            fused_features = torch.cat((audio_features, image_features), dim=1)
            output = self.fc_fusion(fused_features)
        else:
            raise ValueError("Invalid mode or missing input data for the selected mode.")

        return output
