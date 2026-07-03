# GeoEF-Aff
<img width="1244" height="783" alt="3c82469a-bdfe-40dd-a03e-a8c755f4f116" src="https://github.com/user-attachments/assets/ab6923d3-d558-404c-a37d-368eb90a021a" />


## Installation

Using pip:

```bash
pip install -r requirements.txt
```

Or using the captured conda environment:

```bash
conda env create -f env.yaml
conda activate affinity_py310
```

FoldX is not installed by these files. Place the FoldX executable at the path
configured by `FOLDX_PATH` or `FOLDX5_PATH`.

## Quick Start

From the project root:

```bash
python train_3fold_cv.py --split-only
```

Four-GPU sample-level 15-epoch training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  foldx_3feature_retrain_3fold_cv_github/train_3fold_cv.py \
  --split-mode sample \
  --epochs 15 \
  --save-all-epochs \
  --batch-size-per-rank 2 \
  --gradient-accumulation-steps 36 \
  --num-workers-per-rank 6 \
  --output-dir three_fold_results_sample_15epoch
```
## Case Study Testing

`run_case_study.py` is a convenience wrapper for running a trained checkpoint on
small case-study mutation tables. It currently supports two tasks:

- `rbd_ddg`: 6M0J RBD mutation regression.
- `antibody_opt`: 7FAE RBD-Fv antibody candidate ranking.

From the project root, run one checkpoint on the bundled RBD case study:

```bash
CUDA_VISIBLE_DEVICES=0 \
python run_case_study.py rbd_ddg \
  --ckpt /path/to/checkpoint.pth \
  --out-dir casestudy/results/rbd_ddg_example \
  --skip-plots
```

Run the antibody optimization case study:

```bash
CUDA_VISIBLE_DEVICES=0 \
python run_case_study.py antibody_opt \
  --ckpt /path/to/checkpoint.pth \
  --out-dir casestudy/results/antibody_opt_example \
  --skip-plots
```

Default input files are expected at:

- `casestudy/6M0J.pdb` and `casestudy/DDG_6m0j.csv` for `rbd_ddg`.
- `casestudy/7FAE_RBD_Fv.pdb` and `data/7FAE_RBD_Fv_mutation.yml` for
  `antibody_opt`.


## best_model Checkpoint

Model weights are not committed to this repository because checkpoint files are
large. The pretrained checkpoint used in the case-study examples can be
downloaded from Google Drive:

[Download best_model checkpoint](https://drive.google.com/file/d/1Oee9DWvcLEsSFz6OTXs6yUYA6QGp0-WY/view?usp=drive_link)
