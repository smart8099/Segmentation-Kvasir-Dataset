from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="outputs/best.pth")
    parser.add_argument("--out", default="outputs/exported.pth")
    args = parser.parse_args()

    model = models.segmentation.fcn_resnet50(weights=None, num_classes=2)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    torch.save(model.state_dict(), Path(args.out))
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
