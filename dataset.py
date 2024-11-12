# dataset.py
import os
import random
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from preprocess import preprocess_audio, prepare_audio_tensor

class MultimodalDataset(Dataset):
    def __init__(self, audio_dir, image_dir, transform=None):
        self.audio_files = []
        self.image_files = []
        self.labels = []

        # 标签映射
        self.labels_map = {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
            'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7,
            'ship': 8, 'truck': 9
        }

        for category in os.listdir(audio_dir):
            audio_category_dir = os.path.join(audio_dir, category)
            image_category_dir = os.path.join(image_dir, category)

            audio_files = [os.path.join(audio_category_dir, f) for f in os.listdir(audio_category_dir)]
            image_files = [os.path.join(image_category_dir, f) for f in os.listdir(image_category_dir)]

            min_count = min(len(audio_files), len(image_files))
            audio_files = random.sample(audio_files, min_count)
            image_files = random.sample(image_files, min_count)

            self.audio_files.extend(audio_files)
            self.image_files.extend(image_files)
            self.labels.extend([category] * min_count)

        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # 获取标签并转换为数值编码
        label_str = self.labels[idx]
        label = self.labels_map[label_str]
        label = torch.tensor(label, dtype=torch.long)

        # 同时读取音频和图像数据
        audio_path = self.audio_files[idx]
        log_mel_spectrogram = preprocess_audio(audio_path)
        audio_data = prepare_audio_tensor(log_mel_spectrogram)

        image_path = self.image_files[idx]
        image_data = Image.open(image_path).convert('RGB')
        image_data = self.transform(image_data)

        return audio_data, image_data, label  # 返回音频数据、图像数据和标签
