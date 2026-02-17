from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models
import yaml

from data import SegmentationDataset


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_split(output_dir: Path):
    def _load(name: str):
        with open(output_dir / f"{name}.json", "r") as f:
            return json.load(f)
    return _load("train"), _load("val"), _load("test")


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("seg_eval")
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


def classwise_intersection_union(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_labels = preds.argmax(dim=1)
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)

    for class_id in range(num_classes):
        pred_mask = pred_labels == class_id
        target_mask = targets == class_id
        intersections[class_id] += (pred_mask & target_mask).sum().item()
        unions[class_id] += (pred_mask | target_mask).sum().item()
    return intersections, unions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="outputs/best.pth")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    split_dir = Path(cfg["output_dir"]) / "splits"
    _, _, test_items = load_split(split_dir)

    num_classes = int(cfg.get("num_classes", 2))
    include_background = bool(cfg.get("include_background_in_metric", False))
    binarize_masks = bool(cfg.get("binarize_masks", num_classes <= 2))
    class_names = cfg.get("class_names")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "eval.log")

    if class_names and len(class_names) != num_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) must match num_classes ({num_classes})"
        )
    if not class_names:
        class_names = [f"class_{i}" for i in range(num_classes)]

    logger.info("eval_start config=%s weights=%s", args.config, args.weights)
    logger.info(
        "eval_setup test_samples=%d num_classes=%d include_background=%s",
        len(test_items),
        num_classes,
        include_background,
    )

    test_ds = SegmentationDataset(
        test_items, img_size=cfg["img_size"], binarize_masks=binarize_masks
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.segmentation.fcn_resnet50(weights=None, num_classes=num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    iou_total = 0.0
    count = 0
    global_intersection = torch.zeros(num_classes, dtype=torch.float64)
    global_union = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]
            iou_total += compute_iou(
                outputs,
                masks,
                num_classes=num_classes,
                include_background=include_background,
            )
            inter, union = classwise_intersection_union(outputs.cpu(), masks.cpu(), num_classes)
            global_intersection += inter
            global_union += union
            count += 1
    mean_iou = iou_total / max(count, 1)

    start_class = 0 if include_background else 1
    class_iou = {}
    for class_id in range(start_class, num_classes):
        union = global_union[class_id].item()
        iou = 0.0 if union == 0 else (global_intersection[class_id].item() / union)
        class_iou[class_names[class_id]] = iou

    logger.info("test_iou=%.4f", mean_iou)
    print(f"test_iou={mean_iou:.4f}")
    print("per_class_iou:")
    for name, iou in class_iou.items():
        logger.info("class_iou %s=%.4f", name, iou)
        print(f"  {name}: {iou:.4f}")

    metrics = {
        "test_iou": mean_iou,
        "per_class_iou": class_iou,
        "num_test_batches": count,
        "num_test_samples": len(test_items),
        "weights": str(args.weights),
    }
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("saved_metrics=%s", output_dir / "eval_metrics.json")


if __name__ == "__main__":
    main()
