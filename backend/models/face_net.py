"""Compact CNN for FER-2013 (48x48 grayscale -> 7 emotions).

Deliberately small so it runs fast on CPU on the weak deployment device. Imported
lazily (only by training and by CnnFaceEmotionModel), never at backend startup, so
the rest of the API stays torch-free.

Shared by training and inference so the saved state_dict always matches.
"""
import torch
import torch.nn as nn

INPUT_SIZE = 48
# ImageNet-style single-channel normalisation (FER grayscale).
NORM_MEAN = 0.5
NORM_STD = 0.5


class FaceNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 24
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 6
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),  # 3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def preprocess_gray(face_48: "list | object") -> "torch.Tensor":
    """Normalise a 48x48 grayscale array (values 0-255) to a [1,1,48,48] tensor."""
    t = torch.as_tensor(face_48, dtype=torch.float32)
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    t = (t / 255.0 - NORM_MEAN) / NORM_STD
    return t
