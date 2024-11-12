# preprocess.py
import librosa
import numpy as np
import torch
from torchvision import transforms

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(y))
    mel_spectrogram = librosa.feature.melspectrogram(S=stft**2, sr=sr, n_mels=64, fmax=8000)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

def prepare_audio_tensor(log_mel_spectrogram):
    # 调整大小并转换为Tensor
    log_mel_spectrogram = np.resize(log_mel_spectrogram, (32, 32))
    return transforms.ToTensor()(log_mel_spectrogram)
