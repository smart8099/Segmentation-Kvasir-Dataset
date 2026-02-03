from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


IMG_DIR_NAMES = {"images", "image", "imgs", "img", "original"}
MASK_DIR_NAMES = {"masks", "mask", "groundtruth", "ground truth", "gt"}


def find_dir(root: Path, names: set[str]) -> Path | None:
    candidates = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in names:
            candidates.append(p)
    if not candidates:
        return None
    # Prefer PNG subfolders when both TIF and PNG exist.
    candidates.sort(key=lambda p: ("png" not in str(p).lower(), len(p.parts)))
    return candidates[0]


def ensure_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def prepare_dataset(zip_path: Path, out_root: Path, name: str) -> None:
    out_dir = out_root / name
    if not out_dir.exists():
        extract_zip(zip_path, out_dir)

    images_dir = find_dir(out_dir, IMG_DIR_NAMES)
    masks_dir = find_dir(out_dir, MASK_DIR_NAMES)
    if images_dir is None or masks_dir is None:
        raise FileNotFoundError(f"Could not find images/masks in {out_dir}")

    ensure_link_or_copy(images_dir, out_dir / "images")
    ensure_link_or_copy(masks_dir, out_dir / "masks")
    print(f"prepared: {name} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    datasets = {
        "CVC-300": "CVC-300.zip",
        "CVC-ClinicDB": "cvc-clinicdb.zip",
        "CVC-ColonDB": "CVC-ColonDB.zip",
        "ETIS-LaribPolypDB": "ETIS-LaribPolypDB.zip",
    }

    for name, filename in datasets.items():
        zip_path = data_dir / filename
        if not zip_path.exists():
            print(f"skip (missing): {zip_path}")
            continue
        prepare_dataset(zip_path, data_dir, name)

    # Kvasir-SEG uses a different layout; reuse its dedicated prep if present.
    kvasir_zip = data_dir / "kvasir-seg.zip"
    kvasir_out = data_dir / "kvasir-seg"
    if kvasir_zip.exists() and not kvasir_out.exists():
        from pathlib import Path as _Path
        import importlib.util

        script_path = _Path(__file__).parent / "prepare_kvasir_seg.py"
        if script_path.exists():
            spec = importlib.util.spec_from_file_location("prepare_kvasir_seg", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            module.main = module.main  # keep lint calm
            os.system(f"python {script_path} --zip {kvasir_zip} --out {kvasir_out}")
        else:
            print("skip: prepare_kvasir_seg.py not found")


if __name__ == "__main__":
    main()
