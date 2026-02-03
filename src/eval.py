from __future__ import annotations

import argparse
import json
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


def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds = preds.argmax(dim=1)
    preds = (preds > 0).long()
    targets = (targets > 0).long()
    intersection = (preds & targets).sum().item()
    union = (preds | targets).sum().item()
    return intersection / max(union, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="outputs/best.pth")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    split_dir = Path(cfg["output_dir"]) / "splits"
    _, _, test_items = load_split(split_dir)

    test_ds = SegmentationDataset(test_items, img_size=cfg["img_size"])
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.segmentation.fcn_resnet50(weights=None, num_classes=2)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    iou_total = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)["out"]
            iou_total += compute_iou(outputs, masks)
            count += 1
    mean_iou = iou_total / max(count, 1)
    print(f"test_iou={mean_iou:.4f}")


if __name__ == "__main__":
    main()
