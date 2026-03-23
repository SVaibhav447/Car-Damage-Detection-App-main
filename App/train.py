import os
import json
import shutil
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from collections import Counter
from model_helper import CarClassifierResNet

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH  = r"C:\Users\FRIDAY\Desktop\College\Project\Car-Damage-Detection-App-main\Car-Damage-Detection-App-main\image\image"  # ← change this to your local folder path
PREPARED_DIR  = "dataset_clf"
CHECKPOINT    = "detection_model.pth"
NUM_CLASSES   = 8
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-4
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.10
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


VEHIDE_CLASSES = [
    "dent",
    "scratch",
    "broken_glass",
    "lost_parts",
    "punctured",
    "torn",
    "broken_lights",
    "non_damaged"
]

random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# STEP 1 — Download dataset via kagglehub
# ─────────────────────────────────────────────
# def download_dataset():
#     print("Downloading VehiDE dataset...")
#     path = kagglehub.dataset_download(
#         "hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection"
#     )
#     print(f"Downloaded to: {path}")
#     return Path(path)


# ─────────────────────────────────────────────
# STEP 2 — Parse COCO annotations and build
#           an ImageFolder-compatible tree
#
# VehiDE structure:
#   <root>/
#     images/       ← all JPGs live here
#     annotations/
#       instances_train.json
#       instances_val.json   (may not exist — we split manually)
#
# Strategy: for each image, find its dominant damage class
# from the COCO annotations and copy it into:
#   dataset_clf/train/<class>/img.jpg
#   dataset_clf/val/<class>/img.jpg
#   dataset_clf/test/<class>/img.jpg
# ─────────────────────────────────────────────
def build_imagefolder(dataset_root: Path):
    if Path(PREPARED_DIR).exists():
        print(f"'{PREPARED_DIR}' already exists — skipping preparation.")
        return

    print("Building ImageFolder from VIA annotations...")

    ann_files = {
        "train": dataset_root / "0train_via_annos.json",
        "val":   dataset_root / "0val_via_annos.json"
    }

    for split, ann_file in ann_files.items():
        if not ann_file.exists():
            raise FileNotFoundError(f"Cannot find {ann_file}")

        with open(ann_file, encoding="utf-8") as f:
            via_data = json.load(f)

        print(f"{split}: {len(via_data)} images found")

        for fname, entry in via_data.items():
            regions = entry.get("regions", [])

            # Collect all class labels from all regions in this image
            labels = []
            for region in regions:
                label = region.get("class", "").strip().lower()
                if label:
                    labels.append(label)

            # Pick dominant class; fall back to "non_damaged" if no regions
            if not labels:
                dominant = "non_damaged"
            else:
                dominant = Counter(labels).most_common(1)[0][0]

            # Find the image — try direct path first, then recursive search
            src = dataset_root / fname
            if not src.exists():
                matches = list(dataset_root.rglob(fname))
                if not matches:
                    matches = list(dataset_root.parent.rglob(fname))
                    continue
                src = matches[0]

            dst_dir = Path(PREPARED_DIR) / split / dominant
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)

    # Carve out a test split from train (10%)
    print("\nCreating test split from train (10%)...")
    train_dir = Path(PREPARED_DIR) / "train"
    test_dir  = Path(PREPARED_DIR) / "test"
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*"))
        random.shuffle(images)
        n_test = max(1, int(len(images) * 0.10))
        for img in images[:n_test]:
            dst = test_dir / class_dir.name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img), str(dst))

    # Print class distribution
    print("\nFinal dataset structure:")
    for split in ["train", "val", "test"]:
        split_path = Path(PREPARED_DIR) / split
        if not split_path.exists():
            continue
        for cls in sorted(split_path.iterdir()):
            if cls.is_dir():
                count = len(list(cls.glob("*")))
                print(f"  {split}/{cls.name}: {count} images")
# ─────────────────────────────────────────────
# STEP 3 — DataLoaders
# ─────────────────────────────────────────────
def get_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # If val folder missing or empty, carve 15% out of train
    val_path = Path(f"{PREPARED_DIR}/val")
    if not val_path.exists() or not any(val_path.iterdir()):
        print("Val folder missing — creating from 15% of train...")
        train_path = Path(f"{PREPARED_DIR}/train")
        for class_dir in sorted(train_path.iterdir()):
            if not class_dir.is_dir():
                continue
            images = list(class_dir.glob("*"))
            random.shuffle(images)
            n_val = max(1, int(len(images) * 0.15))
            for img in images[:n_val]:
                dst = val_path / class_dir.name / img.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img), str(dst))
        print("Val split created.")

    train_ds = datasets.ImageFolder(f"{PREPARED_DIR}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{PREPARED_DIR}/val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(f"{PREPARED_DIR}/test",  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")
    return train_loader, val_loader, test_loader, train_ds.classes


# ─────────────────────────────────────────────
# STEP 4 — Class-weighted loss to handle imbalance
# ─────────────────────────────────────────────
def get_class_weights(train_loader, num_classes):
    counts = torch.zeros(num_classes)
    for _, labels in train_loader.dataset.samples:
        counts[labels] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights.to(DEVICE)


# ─────────────────────────────────────────────
# STEP 5 — Train one epoch
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# STEP 6 — Evaluate
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# STEP 7 — Full training loop
# ─────────────────────────────────────────────
def train():
    torch.backends.cudnn.benchmark = True
    train_loader, val_loader, test_loader, classes = get_loaders()
    num_classes = len(classes)
    print(f"\nTraining with {num_classes} classes on {DEVICE}\n")

    model = CarClassifierResNet(num_classes=num_classes).to(DEVICE)

    class_weights = get_class_weights(train_loader, num_classes)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader,   criterion)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Saved best model (val acc: {val_acc:.4f})")

    # Final test evaluation
    print("\n─── Test Evaluation ───")
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")

    # Save class names so server knows label order
    with open("class_names.json", "w") as f:
        json.dump(classes, f)
    print(f"\nClass names saved to class_names.json: {classes}")
    print(f"Model saved to {CHECKPOINT}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    dataset_root = Path(DATASET_PATH)
    build_imagefolder(dataset_root)
    train()