"""
Instructor reference: Main file for training YOLOv1 on the BUS dataset (single class, B=1, C=1)

Assumptions:
- model.py contains Yolov1(split_size, num_boxes, num_classes)
- dataset.py contains BusDataset(csv_file, img_dir, label_dir, S, B, C, transform)
- loss.py contains YoloLoss(S, B, C)
- train.csv / val.csv each have two columns:
    col0 = image filename, col1 = label filename
"""

import os
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
# Hyperparameters / Config
# -----------------------------
SEED = 123
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
BATCH_SIZE = 16
EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = False

# YOLO settings for this project
S = 7
B = 1
C = 1  # bus only

# Paths (adjust to your project)
DATA_ROOT = "bus_dataset"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV = os.path.join(DATA_ROOT, "val.csv")
IMG_DIR = os.path.join(DATA_ROOT, "images")
LABEL_DIR = os.path.join(DATA_ROOT, "labels")

LOAD_MODEL = False
LOAD_MODEL_FILE = "bus_overfit.pth.tar"


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism (good for teaching/debugging)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Compose:
    """
    Compose that keeps bboxes unchanged. Works for image-only transforms
    like Resize and ToTensor.
    """

    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, img, bboxes):
        for t in self.transforms_list:
            img = t(img)
        return img, bboxes


def build_transforms(img_size: int = 448):
    return Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )


def build_dataloaders(
    train_csv: str,
    val_csv: str,
    img_dir: str,
    label_dir: str,
    transform,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    train_dataset = BusDataset(
        csv_file=train_csv,
        img_dir=img_dir,
        label_dir=label_dir,
        S=S,
        B=B,
        C=C,
        transform=transform,
    )
    val_dataset = BusDataset(
        csv_file=val_csv,
        img_dir=img_dir,
        label_dir=label_dir,
        S=S,
        B=B,
        C=C,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


def build_model(device: str):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(device)
    return model


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(train_loader, model, optimizer, loss_fn, device: str):
    model.train()
    loop = tqdm(train_loader, leave=True)
    losses = []

    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loop.set_postfix(train_loss=loss.item())

    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def eval_one_epoch(val_loader, model, loss_fn, device: str):
    model.eval()
    losses = []

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        loss = loss_fn(preds, y)
        losses.append(loss.item())

    return float(sum(losses) / max(1, len(losses)))


def maybe_save_checkpoint(model, optimizer, filename: str):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def maybe_load_checkpoint(model, optimizer, filename: str, device: str):
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])



import random
import csv


def create_train_val_csv(
    dataset_root: str,
    train_ratio: float = 0.8,
    seed: int = 123,
):
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")

    images = sorted(os.listdir(images_dir))
    labels = set(os.listdir(labels_dir))

    # keep only images that have labels
    pairs = []
    for img in images:
        label = os.path.splitext(img)[0] + ".txt"
        if label in labels:
            pairs.append((img, label))

    if len(pairs) == 0:
        raise RuntimeError("No matching image/label pairs found.")

    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_csv = os.path.join(dataset_root, "train.csv")
    val_csv = os.path.join(dataset_root, "val.csv")

    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(train_pairs)

    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(val_pairs)

    print(f"Created {train_csv} ({len(train_pairs)} samples)")
    print(f"Created {val_csv} ({len(val_pairs)} samples)")

# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(SEED)

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        create_train_val_csv(DATA_ROOT, train_ratio=0.8, seed=SEED)

    transform = build_transforms(img_size=448)
    train_loader, val_loader = build_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = build_model(DEVICE)
    optimizer = build_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    loss_fn = YoloLoss(S=S, B=B, C=C)

    if LOAD_MODEL and os.path.exists(LOAD_MODEL_FILE):
        maybe_load_checkpoint(model, optimizer, LOAD_MODEL_FILE, DEVICE)
        print(f"Loaded checkpoint: {LOAD_MODEL_FILE}")

    best_val = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
        val_loss = eval_one_epoch(val_loader, model, loss_fn, DEVICE)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Save best model by validation loss (simple + robust)
        if val_loss < best_val:
            best_val = val_loss
            maybe_save_checkpoint(model, optimizer, LOAD_MODEL_FILE)
            print(f"Saved checkpoint (best val so far): {LOAD_MODEL_FILE} (val={best_val:.4f})")


if __name__ == "__main__":
    main()
