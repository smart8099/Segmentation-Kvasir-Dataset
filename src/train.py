from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
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


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    per_class_dice = (2.0 * intersection + eps) / (cardinality + eps)

    start_class = 0 if include_background else 1
    return 1.0 - per_class_dice[start_class:].mean()


def collect_mask_histograms(
    items,
    num_classes: int,
    binarize_masks: bool,
) -> tuple[np.ndarray, list[set[int]]]:
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    sample_class_sets: list[set[int]] = []

    for _, mask_path in items:
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)
        if binarize_masks:
            mask = (mask > 0).astype(np.int64)
        else:
            mask = np.clip(mask, 0, num_classes - 1)

        binc = np.bincount(mask.reshape(-1), minlength=num_classes)
        pixel_counts += binc
        present = set(np.where(binc > 0)[0].tolist())
        sample_class_sets.append(present)

    return pixel_counts, sample_class_sets


def build_class_weights(
    pixel_counts: np.ndarray,
    num_classes: int,
    include_background: bool,
) -> torch.Tensor:
    counts = pixel_counts.astype(np.float64).copy()
    if not include_background and num_classes > 1:
        counts[0] = 0.0

    positive = counts > 0
    if not np.any(positive):
        return torch.ones(num_classes, dtype=torch.float32)

    total = counts[positive].sum()
    weights = np.ones(num_classes, dtype=np.float64)
    weights[positive] = total / (len(np.where(positive)[0]) * counts[positive])
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def build_foreground_sampler_weights(
    sample_class_sets: list[set[int]],
    pixel_counts: np.ndarray,
    num_classes: int,
    strength: float,
) -> torch.Tensor:
    class_scores = np.zeros(num_classes, dtype=np.float64)
    positive = pixel_counts > 0
    class_scores[positive] = 1.0 / np.sqrt(pixel_counts[positive].astype(np.float64))
    if num_classes > 1:
        class_scores[0] = 0.0

    sample_weights = []
    for present in sample_class_sets:
        fg_present = [c for c in present if c > 0]
        if not fg_present:
            sample_weights.append(1.0)
            continue
        rare_score = float(max(class_scores[c] for c in fg_present))
        sample_weights.append(1.0 + strength * rare_score * 1000.0)
    return torch.tensor(sample_weights, dtype=torch.double)


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
    include_background_in_loss = bool(cfg.get("include_background_in_loss", False))
    ce_weight_coef = float(cfg.get("ce_weight_coef", 1.0))
    dice_weight_coef = float(cfg.get("dice_weight_coef", 1.0 if num_classes > 2 else 0.0))
    use_class_weights = bool(cfg.get("use_class_weights", num_classes > 2))
    use_foreground_aware_sampling = bool(cfg.get("foreground_aware_sampling", num_classes > 2))
    fg_sampling_strength = float(cfg.get("fg_sampling_strength", 1.0))

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

    pixel_counts, sample_class_sets = collect_mask_histograms(
        train_items,
        num_classes=num_classes,
        binarize_masks=binarize_masks,
    )
    class_weights = build_class_weights(
        pixel_counts,
        num_classes=num_classes,
        include_background=include_background_in_loss,
    )
    logger.info("pixel_counts=%s", pixel_counts.tolist())
    logger.info("class_weights=%s", [round(float(x), 4) for x in class_weights.tolist()])

    sampler = None
    if use_foreground_aware_sampling:
        sample_weights = build_foreground_sampler_weights(
            sample_class_sets,
            pixel_counts=pixel_counts,
            num_classes=num_classes,
            strength=fg_sampling_strength,
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        logger.info(
            "foreground_aware_sampling enabled strength=%.3f weight_min=%.4f weight_max=%.4f",
            fg_sampling_strength,
            float(sample_weights.min().item()),
            float(sample_weights.max().item()),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
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

    ce_weights_tensor = class_weights.to(device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=ce_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    logger.info(
        "loss_setup ce_weight_coef=%.3f dice_weight_coef=%.3f use_class_weights=%s",
        ce_weight_coef,
        dice_weight_coef,
        use_class_weights,
    )

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
            ce_loss = criterion(outputs, masks)
            d_loss = (
                dice_loss(
                    outputs,
                    masks,
                    num_classes=num_classes,
                    include_background=include_background_in_loss,
                )
                if dice_weight_coef > 0
                else torch.tensor(0.0, device=device)
            )
            loss = ce_weight_coef * ce_loss + dice_weight_coef * d_loss
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
