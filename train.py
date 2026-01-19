"""
STUDENT FILE: Training YOLOv1 on the BUS dataset (single class)

Context:
- You are training a simplified YOLOv1 detector.
- The dataset contains exactly one object per image.
- The loss, dataset, and model architecture are provided.

Your task is to complete a small number of core functions
that define how training happens.
"""

import os
import csv
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import Yolov1
from dataset import BusDataset
from loss import YoloLoss


# -----------------------------
# Configuration
# -----------------------------
SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
BATCH_SIZE = 16
EPOCHS = 20
NUM_WORKERS = 2
PIN_MEMORY = False

# YOLO parameters
S = 7
B = 1
C = 1
IMG_SIZE = 448

# Paths
DATA_ROOT = "bus_dataset"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV = os.path.join(DATA_ROOT, "val.csv")
IMG_DIR = os.path.join(DATA_ROOT, "images")
LABEL_DIR = os.path.join(DATA_ROOT, "labels")

SAVE_FILE = "bus_best.pth.tar"


# -----------------------------
# Instructor-provided utilities
# -----------------------------
def set_seed(seed: int):
    """
    Ensures reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_train_val_csv(dataset_root: str, train_ratio: float = 0.8, seed: int = 123):
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")

    images = sorted(os.listdir(images_dir))
    labels = set(os.listdir(labels_dir))

    pairs = []
    for img in images:
        label = os.path.splitext(img)[0] + ".txt"
        if label in labels:
            pairs.append((img, label))

    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    with open(os.path.join(dataset_root, "train.csv"), "w", newline="") as f:
        csv.writer(f).writerows(train_pairs)

    with open(os.path.join(dataset_root, "val.csv"), "w", newline="") as f:
        csv.writer(f).writerows(val_pairs)


class Compose:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, img, bboxes):
        for t in self.transforms_list:
            img = t(img)
        return img, bboxes


def build_transforms(img_size: int):
    return Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def build_dataloaders(transform):
    train_ds = BusDataset(TRAIN_CSV, IMG_DIR, LABEL_DIR, S=S, B=B, C=C, transform=transform)
    val_ds = BusDataset(VAL_CSV, IMG_DIR, LABEL_DIR, S=S, B=B, C=C, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader


def build_optimizer(model):
    return optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def save_checkpoint(model, optimizer, filename):
    torch.save(
        {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
        filename,
    )


# ============================================================
# ======================== STUDENT TODOs =====================
# ============================================================

def build_model():
    """
    TODO:
    Construct the YOLOv1 model used in this project.

    Consider:
    - the grid size
    - number of bounding boxes per cell
    - number of object classes
    - moving the model to the correct device
    """
    raise NotImplementedError


def train_one_epoch(train_loader, model, optimizer, loss_fn):
    """
    TODO:
    Run one full training epoch.

    You should:
    - place the model in training mode
    - loop over the training data
    - compute predictions and loss
    - update model parameters
    - return the average training loss
    """
    raise NotImplementedError


@torch.no_grad()
def eval_one_epoch(val_loader, model, loss_fn):
    """
    TODO:
    Evaluate the model on the validation set.

    You should:
    - place the model in evaluation mode
    - compute the loss without updating parameters
    - return the average validation loss
    """
    raise NotImplementedError


# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        create_train_val_csv(DATA_ROOT, train_ratio=0.8, seed=SEED)

    set_seed(SEED)

    transform = build_transforms(IMG_SIZE)
    train_loader, val_loader = build_dataloaders(transform)

    model = build_model()
    optimizer = build_optimizer(model)
    loss_fn = YoloLoss(S=S, B=B, C=C)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        val_loss = eval_one_epoch(val_loader, model, loss_fn)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, SAVE_FILE)
            print(f"Checkpoint saved ({SAVE_FILE})")


if __name__ == "__main__":
    main()
