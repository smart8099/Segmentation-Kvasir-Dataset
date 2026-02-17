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

### Synapse (LAS) -> 2D segmentation slices
For Synapse, preprocess `.nii.gz` volumes into 2D PNG `images/` + `masks/` with case-level splits:
```
python scripts/prepare_synapse_seg.py \
  --synapse-root /Users/abdulbasit/Documents/phd_lifetime/endo_agent_project/DATASET_Synapse \
  --task Task002_Synapse \
  --out-root data/synapse_seg \
  --split-out outputs_synapse/splits \
  --axis axial
```
This keeps LAS orientation (no canonical reorientation), preserves class IDs `0..13` in masks, and avoids patient leakage by splitting at case level.

## Split
```
python src/split.py --config configs/default.yaml
```

## Train
```
python src/train.py --config configs/default.yaml
```

For Synapse:
```
python src/train.py --config configs/synapse_las.yaml
```

## Eval
```
python src/eval.py --config configs/default.yaml --weights outputs/best.pth
```

For Synapse:
```
python src/eval.py --config configs/synapse_las.yaml --weights outputs_synapse/best.pth
```

## Slurm (GPU cluster)
Use `slurm_train.sh` as a template. Update `REPO_DIR` and the zip filename before submitting.
