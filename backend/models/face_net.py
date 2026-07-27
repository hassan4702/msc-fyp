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


# ---------------------------------------------------------------------------
# ResNet-18 backbone (the current model; FaceNet above is the from-scratch
# baseline it replaced, kept because §9 reports both).
# ---------------------------------------------------------------------------
RESNET_INPUT_SIZE = 224
# ImageNet statistics — required, since the backbone's pretrained filters expect them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet18(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ResNet-18 with a fresh `num_classes` head."""
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def preprocess_resnet(face_gray: "object") -> "torch.Tensor":
    """Grayscale face crop (any size, 0-255) -> normalised [1,3,224,224] tensor.

    Grayscale is replicated across the three channels; the backbone is pretrained on
    RGB and this is the standard way to feed it single-channel input.
    """
    import cv2
    import numpy as np

    arr = np.asarray(face_gray, dtype=np.uint8)
    resized = cv2.resize(arr, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    t = torch.as_tensor(resized, dtype=torch.float32).div_(255.0)
    t = t.unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)
