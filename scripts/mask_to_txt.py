from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Export mask image pixels to a text file")
    parser.add_argument("--mask", required=True, help="Path to mask image")
    parser.add_argument("--out", default="", help="Output .txt path (default: <mask_stem>.txt)")
    parser.add_argument("--delimiter", default=" ", help="Value delimiter in txt (default: space)")
    parser.add_argument("--fmt", default="%d", help="NumPy savetxt format string (default: %%d)")
    args = parser.parse_args()

    mask_path = Path(args.mask).expanduser().resolve()
    if not mask_path.exists():
        raise FileNotFoundError(f"mask not found: {mask_path}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else mask_path.with_suffix(".txt")
    )

    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        # If a multi-channel mask is provided, save the first channel.
        arr = arr[..., 0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, arr.astype(np.int64), fmt=args.fmt, delimiter=args.delimiter)

    print(f"saved={out_path}")
    print(f"shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
