import os
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms

from model import Yolov1
from dataset import BusDataset

# --- Config ---
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

S, B, C = 7, 1, 1
IMG_SIZE = 448

DATA_ROOT = "bus_dataset"
CSV_FILE = os.path.join(DATA_ROOT, "val.csv")   # or train.csv
IMG_DIR = os.path.join(DATA_ROOT, "images")
LABEL_DIR = os.path.join(DATA_ROOT, "labels")

CHECKPOINT = "bus_overfit.pth.tar"  # change if different
CONF_THRESH = 0.25

# How many random images to visualize per run
NUM_RANDOM_SAMPLES = 5


def load_checkpoint(model, filename):
    ckpt = torch.load(filename, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model


def decode_one_box_from_cellgrid(pred_grid, S=7, C=1):
    """
    pred_grid: (S, S, C+5) for ONE image, after reshaping model output
    Layout per cell: [class_probs(C)] [conf] [x y w h]
    Returns (conf, x, y, w, h) in normalized image coords by picking max-confidence cell.
    """
    conf_map = pred_grid[..., C]  # (S,S)
    flat_idx = torch.argmax(conf_map).item()
    i, j = flat_idx // S, flat_idx % S
    conf = conf_map[i, j].item()

    x_cell, y_cell, w_cell, h_cell = pred_grid[i, j, C + 1 : C + 5]

    # cell-relative -> image-relative (normalized)
    x = (j + x_cell.item()) / S
    y = (i + y_cell.item()) / S
    w = w_cell.item() / S
    h = h_cell.item() / S

    return conf, x, y, w, h


def decode_gt_from_label_file(label_path):
    """
    Reads raw YOLO label file: "class x y w h" (normalized midpoint)
    Assumes ONE object per image. Returns (x, y, w, h) normalized.
    """
    with open(label_path, "r") as f:
        line = f.readline().strip()

    if not line:
        return None

    parts = line.split()
    if len(parts) < 5:
        return None

    _, x, y, w, h = parts
    return float(x), float(y), float(w), float(h)


def draw_box(ax, x, y, w, h, label, img_h, img_w, color="red", linewidth=2):
    """
    x,y,w,h are normalized midpoint coords (0..1).
    Draw box in pixel coordinates on ax.
    """
    # midpoint -> top-left (normalized)
    x1 = x - w / 2
    y1 = y - h / 2

    # normalized -> pixels
    x1_px = x1 * img_w
    y1_px = y1 * img_h
    w_px = w * img_w
    h_px = h * img_h

    rect = patches.Rectangle(
        (x1_px, y1_px),
        w_px,
        h_px,
        fill=False,
        linewidth=linewidth,
        edgecolor=color,
    )
    ax.add_patch(rect)
    ax.text(
        x1_px,
        y1_px,
        label,
        fontsize=10,
        color=color,
        verticalalignment="bottom",
    )


def plot_prediction_on_sample(model, dataset, index=0, conf_thresh=0.25):
    """
    Plots:
      - Ground truth from RAW label file (most reliable sanity check)
      - Prediction from model (best-confidence cell)
    """
    img, _ = dataset[index]  # img: (3, H, W)

    # Get the raw label path from the CSV row (dataset already loaded it)
    label_filename = dataset.annotations.iloc[index, 1]
    label_path = os.path.join(LABEL_DIR, label_filename)

    # Prepare image for model (batch)
    x = img.unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred = model(x)  # (1, S*S*(C+5))
    pred_grid = pred.reshape(1, S, S, C + 5)[0].cpu()

    conf, px, py, pw, ph = decode_one_box_from_cellgrid(pred_grid, S=S, C=C)

    # Plot image
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_h, img_w = img_np.shape[0], img_np.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(img_np)
    ax.set_axis_off()

    # --- Ground Truth from RAW label file ---
    gt = decode_gt_from_label_file(label_path)
    if gt is not None:
        gx, gy, gw, gh = gt
        draw_box(ax, gx, gy, gw, gh, label="GT", img_h=img_h, img_w=img_w, color="lime", linewidth=2)
    else:
        ax.set_title("No GT found in label file")

    # --- Prediction ---
    # If you want to ALWAYS see the predicted box, set conf_thresh = -1.0
    if conf >= conf_thresh:
        draw_box(ax, px, py, pw, ph, label=f"Pred {conf:.2f}", img_h=img_h, img_w=img_w, color="red", linewidth=2)
    else:
        ax.set_title(f"No Pred >= {conf_thresh} (best conf={conf:.2f})")

    plt.show()


def main():
    # BusDataset expects transform(img, boxes) -> (img, boxes)
    class SimpleCompose:
        def __init__(self, tfs):
            self.tfs = tfs

        def __call__(self, img, bboxes):
            for t in self.tfs:
                img = t(img)
            return img, bboxes

    ds = BusDataset(
        csv_file=CSV_FILE,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=S,
        B=B,
        C=C,
        transform=SimpleCompose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ]
        ),
    )

    model = Yolov1(split_size=S, num_boxes=B, num_classes=C).to(DEVICE)

    if os.path.exists(CHECKPOINT):
        model = load_checkpoint(model, CHECKPOINT)
        print(f"Loaded checkpoint: {CHECKPOINT}")
    else:
        print(f"Checkpoint not found: {CHECKPOINT} (will plot random/untrained predictions)")

    # Pick random indices and plot
    n = len(ds)
    indices = random.sample(range(n), k=min(NUM_RANDOM_SAMPLES, n))

    for idx in indices:
        plot_prediction_on_sample(model, ds, index=idx, conf_thresh=CONF_THRESH)


if __name__ == "__main__":
    main()
