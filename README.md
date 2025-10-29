# UAED on TBM: Original vs Super-Resolution

Edge detection for metallography on the TBM dataset. Two tracks: training and evaluating UAED on the original TBM images and on a super-resolved variant (SR_TBM). 

Pretrained checkpoints and datasets download available on: https://zenodo.org/records/16918388

## Repository layout

```
.
├── checkpoints/
│   ├── uaed_tbm.pth          # trained on TBM_original
│   └── uaed_sr_tbm.pth       # trained on SR_TBM
├── scripts/
│   ├── train.py              # training entry point
│   ├── eval_on_folder.py     # evaluation on an images+masks folder
│   ├── train_original/repro.sh
│   ├── train_sr/repro.sh
│   ├── eval_original_configs/run_command.txt
│   └── eval_sr_configs/run_command.txt
├── TBM_original/             # original-resolution dataset
│   ├── train/{images,images_1024,masks}
│   ├── val/{images,images_1024,masks}
│   └── test/{images,images_1024,masks}
└── SR_TBM/                   # super-resolved dataset
    ├── train/{images,masks}
    ├── val/{images,masks}
    └── test/{images,masks}
```

**Notes**
- `images_1024/` are linearly upscaled originals for parity with SR inputs.
- Filenames in `images/` and `masks/` are one-to-one.

## Environment

Runs under the UAED environment. Follow the setup from the UAED repository and keep the same Python/CUDA stack.

- UAED GitHub: <https://github.com/ZhouCX117/UAED_MuGE>

Minimal extras commonly used here (install after the UAED setup):
```bash
pip install numpy pandas scikit-image imageio matplotlib tqdm
```

## Quick start

### Evaluate the provided checkpoints

**Original TBM**
```bash
python scripts/eval_on_folder.py   --images TBM_original/test/images_1024   --masks  TBM_original/test/masks   --ckpt   checkpoints/uaed_tbm.pth   --out    outputs/tbm_eval
# Or replicate our exact command(s):
cat scripts/eval_original_configs/run_command.txt
```

**SR_TBM**
```bash
python scripts/eval_on_folder.py   --images SR_TBM/test/images   --masks  SR_TBM/test/masks   --ckpt   checkpoints/uaed_sr_tbm.pth   --out    outputs/sr_tbm_eval
# Or:
cat scripts/eval_sr_configs/run_command.txt
```

**Help**
```bash
python scripts/eval_on_folder.py -h
```

**Expected outputs**
- Probability maps, uncertainty maps, thresholded masks, and overlays.
- CSV report with AP, ROC-AUC, thresholded metrics, and a pooled Heyn grain-size estimate.
- PR/ROC curve images.

### Train from scratch

**Original TBM**
```bash
bash scripts/train_original/repro.sh
```

**SR_TBM**
```bash
bash scripts/train_sr/repro.sh
```

**Direct help**
```bash
python scripts/train.py -h
```

**Outputs**
- Checkpoints saved under `checkpoints/` (or a run directory if configured).
- Validation figures similar to eval.
- Optional logs/TensorBoard if enabled in the scripts.

## Data

Folder convention:
```
DATA_ROOT/{train,val,test}/images/*.png
DATA_ROOT/{train,val,test}/masks/*.png
# TBM_original also has {train,val,test}/images_1024/*.png
```
Masks are single-channel PNGs (0/255 or 0/1). Filenames match between `images/` and `masks/`.

## Checkpoints

- `checkpoints/uaed_tbm.pth` — UAED trained on `TBM_original`.
- `checkpoints/uaed_sr_tbm.pth` — UAED trained on `SR_TBM`.

Use `--ckpt` in `eval_on_folder.py` or the training script to select a model.

## Reproducibility

- Use the exact commands in:
  - `scripts/train_original/repro.sh`
  - `scripts/train_sr/repro.sh`
  - `scripts/eval_*_configs/run_command.txt`
- Fix seeds and keep hardware/software constant when possible.

## Acknowledgments

- **UAED** for the uncertainty-aware edge detection framework and training codebase. <https://github.com/ZhouCX117/UAED_MuGE>
- **OSEDiff** for one-step effective diffusion SR used to produce SR_TBM. <https://github.com/cswry/OSEDiff>

## Citation

If this repository helps your work, please cite our accompanying paper(s) and the upstream projects UAED and OSEDiff. 

## License

Apache-2.0

## Contact

Open an issue or pull request with a minimal reproducible example.
