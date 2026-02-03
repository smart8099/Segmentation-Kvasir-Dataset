from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


def find_dir(root: Path, names):
    candidates = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in names:
            candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def ensure_link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    zip_path = Path(args.zip).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"zip not found: {zip_path}")

    extract_zip(zip_path, out_dir)

    images_dir = find_dir(out_dir, {"images"})
    masks_dir = find_dir(out_dir, {"masks", "mask"})
    if images_dir is None or masks_dir is None:
        raise FileNotFoundError("Could not find images/ and masks/ after extraction")

    canonical_images = out_dir / "images"
    canonical_masks = out_dir / "masks"
    ensure_link_or_copy(images_dir, canonical_images)
    ensure_link_or_copy(masks_dir, canonical_masks)

    num_images = len([p for p in canonical_images.rglob("*") if p.is_file()])
    num_masks = len([p for p in canonical_masks.rglob("*") if p.is_file()])
    print(f"images={num_images} masks={num_masks} root={out_dir}")


if __name__ == "__main__":
    main()
