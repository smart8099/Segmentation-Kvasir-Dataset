# Segmentation (Kvasir-SEG)

Standalone repo to train a binary polyp segmentation model on Kvasir-SEG.

## Setup
```
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Prepare data
Place the Kvasir-SEG zip in `data/` and run:
```
python scripts/prepare_kvasir_seg.py --zip data/kvasir-seg.zip --out data/kvasir-seg
```

For multiple datasets, extract each into its own folder under `data/` with
`images/` and `masks/` subfolders.
You can batch-prepare common polyp datasets with:
```
python scripts/prepare_multi_seg.py --data-dir data
```

## Split
```
python src/split.py --config configs/default.yaml
```

## Train
```
python src/train.py --config configs/default.yaml
```

## Eval
```
python src/eval.py --config configs/default.yaml --weights outputs/best.pth
```

## Slurm (GPU cluster)
Use `slurm_train.sh` as a template. Update `REPO_DIR` and the zip filename before submitting.
