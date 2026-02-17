from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image


def parse_split(split: str) -> Dict[str, float]:
    vals = [float(v.strip()) for v in split.split(",")]
    if len(vals) != 3:
        raise ValueError("--split must be train,val,test")
    if any(v < 0 for v in vals):
        raise ValueError("split values must be non-negative")
    total = sum(vals)
    if total <= 0:
        raise ValueError("split values must sum to > 0")
    return {"train": vals[0] / total, "val": vals[1] / total, "test": vals[2] / total}


def load_task_root(synapse_root: Path, task_name: str) -> Path:
    task_root = synapse_root / "unetr_pp_raw" / "unetr_pp_raw_data" / task_name
    if not task_root.exists():
        raise FileNotFoundError(f"Task root not found: {task_root}")
    for d in ["imagesTr", "labelsTr"]:
        if not (task_root / d).exists():
            raise FileNotFoundError(f"Missing {d} under {task_root}")
    return task_root


def load_dataset_meta(task_root: Path) -> dict:
    dataset_json = task_root / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing dataset.json: {dataset_json}")
    with open(dataset_json, "r") as f:
        return json.load(f)


def case_id_from_path(path: Path) -> str:
    base = path.name
    if base.endswith(".nii.gz"):
        base = base[: -len(".nii.gz")]
    elif base.endswith(".nii"):
        base = base[: -len(".nii")]
    return base[:-5] if base.endswith("_0000") else base


def parse_training_pairs(task_root: Path, meta: dict) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    for item in meta.get("training", []):
        image_rel = item["image"].replace("./", "")
        label_rel = item["label"].replace("./", "")
        image_path = task_root / image_rel
        label_path = task_root / label_rel

        # Handle metadata listing imgXXXX.nii.gz while files are imgXXXX_0000.nii.gz.
        if not image_path.exists() and image_path.name.endswith(".nii.gz"):
            alt_name = image_path.name.replace(".nii.gz", "_0000.nii.gz")
            alt_path = image_path.with_name(alt_name)
            if alt_path.exists():
                image_path = alt_path

        if not image_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Missing pair: {image_path} / {label_path}")

        pairs.append((image_path, label_path, case_id_from_path(image_path)))

    if not pairs:
        raise ValueError("No training pairs found in dataset.json")
    return pairs


def split_cases(case_ids: List[str], split_cfg: Dict[str, float], seed: int) -> Dict[str, set[str]]:
    ids = case_ids[:]
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])

    if not train_ids or not val_ids or not test_ids:
        print("warning: one split has zero cases; consider changing --split")

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def choose_split(case_id: str, case_splits: Dict[str, set[str]]) -> str:
    for split_name, members in case_splits.items():
        if case_id in members:
            return split_name
    raise KeyError(f"case_id {case_id} missing from split map")


def window_to_uint8(slice_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    clipped = np.clip(slice_hu, lo, hi)
    scaled = (clipped - lo) / max(hi - lo, 1e-6)
    return (scaled * 255.0).astype(np.uint8)


def axis_to_int(axis: str) -> int:
    lookup = {"sagittal": 0, "coronal": 1, "axial": 2}
    if axis not in lookup:
        raise ValueError(f"unsupported axis: {axis}")
    return lookup[axis]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synapse-root",
        default="/Users/abdulbasit/Documents/phd_lifetime/endo_agent_project/DATASET_Synapse",
        help="Path to DATASET_Synapse root",
    )
    parser.add_argument("--task", default="Task002_Synapse")
    parser.add_argument("--out-root", default="data/synapse_seg")
    parser.add_argument("--split-out", default="outputs_synapse/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test")
    parser.add_argument("--axis", default="axial", choices=["axial", "coronal", "sagittal"])
    parser.add_argument("--window", default="50,400", help="CT window center,width")
    parser.add_argument("--slice-step", type=int, default=1, help="Keep every Nth slice")
    parser.add_argument(
        "--min-foreground-pixels",
        type=int,
        default=200,
        help="Skip slices with fewer than this many non-zero mask pixels",
    )
    parser.add_argument(
        "--keep-background-slices",
        action="store_true",
        help="Keep slices regardless of foreground pixels",
    )
    parser.add_argument(
        "--max-slices-per-case",
        type=int,
        default=0,
        help="If > 0, cap exported slices per case after filtering",
    )
    args = parser.parse_args()

    split_cfg = parse_split(args.split)
    if args.slice_step < 1:
        raise ValueError("--slice-step must be >= 1")

    center_str, width_str = args.window.split(",")
    window_center = float(center_str)
    window_width = float(width_str)

    synapse_root = Path(args.synapse_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    split_out = Path(args.split_out).expanduser().resolve()

    images_out = out_root / "images"
    masks_out = out_root / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    split_out.mkdir(parents=True, exist_ok=True)

    task_root = load_task_root(synapse_root, args.task)
    meta = load_dataset_meta(task_root)
    pairs = parse_training_pairs(task_root, meta)

    case_splits = split_cases([case_id for _, _, case_id in pairs], split_cfg, args.seed)
    axis = axis_to_int(args.axis)

    split_items: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}
    split_case_counts: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()

    total_saved = 0
    total_skipped = 0

    for image_path, label_path, case_id in pairs:
        vol_nii = nib.load(str(image_path))
        seg_nii = nib.load(str(label_path))

        vol_ax = nib.aff2axcodes(vol_nii.affine)
        seg_ax = nib.aff2axcodes(seg_nii.affine)
        orientation_counts["".join(vol_ax)] += 1
        if vol_ax != seg_ax:
            raise ValueError(f"orientation mismatch for {case_id}: image={vol_ax} label={seg_ax}")

        vol = vol_nii.get_fdata(dtype=np.float32)
        seg = seg_nii.get_fdata(dtype=np.float32).astype(np.int16)

        if vol.shape != seg.shape:
            raise ValueError(f"shape mismatch for {case_id}: {vol.shape} vs {seg.shape}")

        split_name = choose_split(case_id, case_splits)
        split_case_counts[split_name] += 1

        saved_for_case = 0
        num_slices = vol.shape[axis]

        for idx in range(0, num_slices, args.slice_step):
            if axis == 2:
                img2d = vol[:, :, idx]
                mask2d = seg[:, :, idx]
            elif axis == 1:
                img2d = vol[:, idx, :]
                mask2d = seg[:, idx, :]
            else:
                img2d = vol[idx, :, :]
                mask2d = seg[idx, :, :]

            foreground = int((mask2d > 0).sum())
            if not args.keep_background_slices and foreground < args.min_foreground_pixels:
                total_skipped += 1
                continue

            img_u8 = window_to_uint8(img2d, center=window_center, width=window_width)
            mask_u8 = np.clip(mask2d, 0, 255).astype(np.uint8)

            stem = f"{case_id}_{args.axis}_{idx:04d}"
            img_out = images_out / f"{stem}.png"
            mask_out = masks_out / f"{stem}.png"

            Image.fromarray(img_u8, mode="L").save(img_out)
            Image.fromarray(mask_u8, mode="L").save(mask_out)

            split_items[split_name].append((str(img_out), str(mask_out)))
            total_saved += 1
            saved_for_case += 1

            if args.max_slices_per_case > 0 and saved_for_case >= args.max_slices_per_case:
                break

    for split_name in ["train", "val", "test"]:
        with open(split_out / f"{split_name}.json", "w") as f:
            json.dump(split_items[split_name], f)

    summary = {
        "task_root": str(task_root),
        "out_root": str(out_root),
        "split_out": str(split_out),
        "orientation_expected": "LAS",
        "orientation_counts": dict(orientation_counts),
        "axis": args.axis,
        "window": {"center": window_center, "width": window_width},
        "slice_step": args.slice_step,
        "min_foreground_pixels": args.min_foreground_pixels,
        "keep_background_slices": bool(args.keep_background_slices),
        "split": split_cfg,
        "num_cases": len(pairs),
        "case_split_counts": dict(split_case_counts),
        "slice_split_counts": {k: len(v) for k, v in split_items.items()},
        "slices_saved": total_saved,
        "slices_skipped": total_skipped,
        "num_classes": 14,
    }
    with open(split_out / "synapse_seg_prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("prepared Synapse segmentation dataset (LAS)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
