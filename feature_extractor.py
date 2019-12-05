import sys
import os

import librosa
import numpy as np
import torch
from torchvision import transforms

from models import CQTTPPNet
from cqt_loader import cut_data_front
from utility import norm


def load_resources():
    model = CQTTPPNet()
    model.load("check_points/best.pth", map_location="cpu")
    model.eval()

    tfms = transforms.Compose(
        [
            lambda x: x.T,
            # lambda x : x-np.mean(x),
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: cut_data_front(x, None),
            lambda x: torch.Tensor(x),
            lambda x: x.permute(1, 0).unsqueeze(0),
        ]
    )

    return {"model": model, "tfms": tfms}


def extract_features(resources, audio_path):
    model = resources["model"]
    tfms = resources["tfms"]

    data, sr = librosa.load(audio_path)
    if len(data) < 1000:
        raise ValueError("Audio is fewer than 1000 samples")

    cqt = np.abs(librosa.cqt(y=data, sr=sr))
    mean_size = 20
    height, length = cqt.shape
    new_cqt = np.zeros((height, int(length / mean_size)), dtype=np.float64)
    for i in range(int(length / mean_size)):
        new_cqt[:, i] = cqt[:, i * mean_size : (i + 1) * mean_size].mean(axis=1)

    transformed = tfms(new_cqt)
    with torch.no_grad():
        _, feature = model(transformed.unsqueeze(0))

    feature = feature.numpy()
    feature = norm(feature)

    return {"vector": feature[0]}
