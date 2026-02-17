from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import yaml

from data import SegmentationDataset, index_pairs, index_pairs_multi, random_split


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_split(output_dir: Path, split):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", split.train), ("val", split.val), ("test", split.test)]:
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(items, f)


def load_split(output_dir: Path):
    def _load(name: str):
        with open(output_dir / f"{name}.json", "r") as f:
            return json.load(f)
    return _load("train"), _load("val"), _load("test")


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
) -> float:
    preds = preds.argmax(dim=1)

    if num_classes <= 2:
        preds = (preds > 0).long()
        targets = (targets > 0).long()
        intersection = (preds & targets).sum().item()
        union = (preds | targets).sum().item()
        return intersection / max(union, 1)

    start_class = 0 if include_background else 1
    ious = []
    for class_id in range(start_class, num_classes):
        pred_mask = preds == class_id
        target_mask = targets == class_id
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        intersection = (pred_mask & target_mask).sum().item()
        ious.append(intersection / union)
    return float(sum(ious) / max(len(ious), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    torch.manual_seed(cfg["seed"])

    split_dir = Path(cfg["output_dir"]) / "splits"
    if (split_dir / "train.json").exists():
        train_items, val_items, _ = load_split(split_dir)
    else:
        data_roots_cfg = cfg.get("data_roots")
        if data_roots_cfg:
            data_roots = [Path(p).expanduser().resolve() for p in data_roots_cfg]
            items = index_pairs_multi(data_roots)
            by_source = {}
            for img, mask, source in items:
                by_source.setdefault(source, []).append((img, mask))
            merged = {"train": [], "val": [], "test": []}
            for source, pairs in by_source.items():
                split = random_split(pairs, seed=cfg["seed"], split=cfg["split"])
                merged["train"].extend(split.train)
                merged["val"].extend(split.val)
                merged["test"].extend(split.test)
            split = type("Split", (), merged)
        else:
            data_root = Path(cfg["data_root"]).resolve()
            pairs = index_pairs(data_root)
            split = random_split(pairs, seed=cfg["seed"], split=cfg["split"])

        save_split(split_dir, split)
        train_items, val_items = split.train, split.val

    num_classes = int(cfg.get("num_classes", 2))
    include_background = bool(cfg.get("include_background_in_metric", False))
    binarize_masks = bool(cfg.get("binarize_masks", num_classes <= 2))

    train_ds = SegmentationDataset(
        train_items, img_size=cfg["img_size"], binarize_masks=binarize_masks
    )
    val_ds = SegmentationDataset(
        val_items, img_size=cfg["img_size"], binarize_masks=binarize_masks
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.segmentation.fcn_resnet50(weights=None, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_iou = 0.0
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        iou_total = 0.0
        count = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)["out"]
                iou_total += compute_iou(
                    outputs,
                    masks,
                    num_classes=num_classes,
                    include_background=include_background,
                )
                count += 1
        mean_iou = iou_total / max(count, 1)

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), output_dir / "best.pth")

        print(
            f"epoch={epoch+1} train_loss={running/ max(len(train_loader),1):.4f} "
            f"val_iou={mean_iou:.4f}"
        )

    torch.save(model.state_dict(), output_dir / "last.pth")


if __name__ == "__main__":
    main()
