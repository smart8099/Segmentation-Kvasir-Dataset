from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data import index_pairs, index_pairs_multi, random_split


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_split(output_dir: Path, split):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in [("train", split.train), ("val", split.val), ("test", split.test)]:
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(items, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    split_dir = Path(cfg["output_dir"]) / "splits"

    data_roots_cfg = cfg.get("data_roots")
    if data_roots_cfg:
        data_roots = [Path(p).expanduser().resolve() for p in data_roots_cfg]
        items = index_pairs_multi(data_roots)
        # split per source to preserve dataset proportions
        by_source = {}
        for img, mask, source in items:
            by_source.setdefault(source, []).append((img, mask))
        split = None
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

    print(f"train={len(split.train)} val={len(split.val)} test={len(split.test)}")


if __name__ == "__main__":
    main()
