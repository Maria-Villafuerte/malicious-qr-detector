"""
CNN puro para clasificar imágenes QR como malignas o benignas.
Entrada: imagen 128×128 en escala de grises.
Salida : probabilidad de ser malware.

Uso:
    python 02_cnn_model.py
    python 02_cnn_model.py --meta_csv data_procesada/qr_full_dataset.csv --epochs 30

Dependencias:
    pip install torch torchvision scikit-learn pandas pillow tqdm
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED     = 42
IMG_SIZE = 128

# 1. DATASET
class QRDataset(Dataset):
    def __init__(self, filepaths: list, labels: list, transform=None):
        self.filepaths = filepaths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img = torch.tensor(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)  # (1, H, W)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label



# 2. ARQUITECTURA
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class QRCNN(nn.Module):
    """
    CNN ligera para clasificación binaria de QR codes.

    Entrada : (B, 1, 128, 128)
    Salida  : (B, 2)  — logits [normal, malware]

    Arquitectura:
        ConvBlock(1  → 32)   128×128 → 64×64
        ConvBlock(32 → 64)    64×64  → 32×32
        ConvBlock(64 → 128)   32×32  → 16×16
        ConvBlock(128→ 256)   16×16  →  8×8
        Conv(256→256) + BN + ReLU     8×8
        AdaptiveAvgPool → 4×4
        Flatten → 4096
        FC(4096→512) → ReLU → Dropout
        FC(512→128)  → ReLU → Dropout
        FC(128→2)
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            # Capa extra sin pooling para más capacidad representacional
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))



# 3. LOOPS DE ENTRENAMIENTO Y EVALUACIÓN
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_probs, all_labels = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1]

        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        roc_auc_score(all_labels, all_probs),
        all_preds,
        all_labels,
    )

# 5. INFERENCIA EN UNA IMAGEN
def predict(image_path: str, model_path: str = "outputs/best_model.pt", device: str = "cpu") -> dict:
    """
    Predice si un QR code es malicioso.

    Ejemplo:
        result = predict("mi_qr.png")
        print(result)  # {"label": "malware", "probability_malware": 0.923}
    """
    dev   = torch.device(device)
    model = QRCNN()
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.to(dev).eval()

    img = Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    img_t = torch.tensor(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)

    with torch.no_grad():
        prob = torch.softmax(model(img_t), dim=1)[0, 1].item()

    return {"label": "malware" if prob >= 0.5 else "normal", "probability_malware": round(prob, 4)}


if __name__ == "__main__":
    main()