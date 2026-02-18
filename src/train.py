from __future__ import annotations

import argparse
import csv
import json
import logging
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


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("seg_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


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
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "train.log")
    logger.info("train_start config=%s", args.config)

    split_dir = output_dir / "splits"
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
    logger.info(
        "train_setup train_samples=%d val_samples=%d num_classes=%d binarize_masks=%s",
        len(train_items),
        len(val_items),
        num_classes,
        binarize_masks,
    )

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
    metrics_csv = output_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_iou", "is_best"])

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

        is_best = False
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), output_dir / "best.pth")
            is_best = True

        epoch_loss = running / max(len(train_loader), 1)
        line = f"epoch={epoch+1} train_loss={epoch_loss:.4f} val_iou={mean_iou:.4f}"
        print(line)
        logger.info("%s%s", line, " [best]" if is_best else "")
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{epoch_loss:.6f}", f"{mean_iou:.6f}", int(is_best)])

    torch.save(model.state_dict(), output_dir / "last.pth")
    logger.info(
        "train_done best_val_iou=%.4f best_ckpt=%s last_ckpt=%s metrics_csv=%s",
        best_iou,
        output_dir / "best.pth",
        output_dir / "last.pth",
        metrics_csv,
    )


if __name__ == "__main__":
    main()
