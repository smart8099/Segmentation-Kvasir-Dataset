from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torchvision.transforms import functional as F


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SplitFiles:
    train: List[Tuple[str, str]]
    val: List[Tuple[str, str]]
    test: List[Tuple[str, str]]


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def find_images_masks(data_root: Path) -> Tuple[Path, Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    if images_dir.exists() and masks_dir.exists():
        return images_dir, masks_dir

    # Walk to find closest images/masks directories
    candidates = []
    for p in data_root.rglob("*"):
        if p.is_dir() and p.name.lower() == "images":
            candidates.append(p)
    if candidates:
        images_dir = sorted(candidates, key=lambda p: len(p.parts))[0]

    candidates = []
    for p in data_root.rglob("*"):
        if p.is_dir() and p.name.lower() in {"masks", "mask"}:
            candidates.append(p)
    if candidates:
        masks_dir = sorted(candidates, key=lambda p: len(p.parts))[0]

    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError("Could not locate images/ and masks/ directories")

    return images_dir, masks_dir


def index_pairs(data_root: Path) -> List[Tuple[str, str]]:
    images_dir, masks_dir = find_images_masks(data_root)
    mask_map = {p.stem: p for p in masks_dir.rglob("*") if p.is_file() and _is_image(p)}

    pairs: List[Tuple[str, str]] = []
    for img in images_dir.rglob("*"):
        if img.is_file() and _is_image(img):
            mask = mask_map.get(img.stem)
            if mask is not None:
                pairs.append((str(img), str(mask)))

    if not pairs:
        raise FileNotFoundError("No image/mask pairs found under data_root")

    pairs.sort()
    return pairs


def random_split(pairs: List[Tuple[str, str]], seed: int, split: dict) -> SplitFiles:
    rng = random.Random(seed)
    items = pairs[:]
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * split["train"])
    n_val = int(n * split["val"])
    n_test = n - n_train - n_val

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    if n_test < 0:
        raise ValueError("Invalid split ratios")

    return SplitFiles(train=train, val=val, test=test)


class SegmentationDataset:
    def __init__(self, items: List[Tuple[str, str]], img_size: int):
        self.items = items
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = F.resize(image, [self.img_size, self.img_size])
        mask = F.resize(mask, [self.img_size, self.img_size], interpolation=F.InterpolationMode.NEAREST)

        image = F.to_tensor(image)
        mask = F.pil_to_tensor(mask).squeeze(0)
        mask = (mask > 0).long()

        return image, mask


def index_pairs_multi(data_roots: List[Path]) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    for root in data_roots:
        images_dir, masks_dir = find_images_masks(root)
        mask_map = {p.stem: p for p in masks_dir.rglob("*") if p.is_file() and _is_image(p)}
        for img in images_dir.rglob("*"):
            if img.is_file() and _is_image(img):
                mask = mask_map.get(img.stem)
                if mask is not None:
                    items.append((str(img), str(mask), root.name))
    if not items:
        raise FileNotFoundError("No image/mask pairs found across data_roots")
    return items
