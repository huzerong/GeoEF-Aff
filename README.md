# GeoEF-Aff

GeoEF-Aff is a mutation-aware multimodal framework for predicting
mutation-induced changes in protein--protein binding affinity. It combines
global and mutation-local ESM-2 representations, RAAD-based interface
geometry, mutation-type features, and WT--Mut--Delta FoldX descriptors.

This release is organized around one primary experiment:

- complex-grouped train/validation/test split;
- fixed ratio of 80:10:10;
- random seed 42 by default;
- frozen primary partition in `splits/fold_01/fold_01_split.json`;
- validation data used for model selection and early stopping;
- test data evaluated only after reloading `best_model.pth`.

Three independently seeded 80:10:10 partition manifests are retained. The
release distributes one final checkpoint, `models/best_model.pth`, rather than
one checkpoint per fold. Details are provided in
[`THREE_FOLD_EVALUATION.md`](THREE_FOLD_EVALUATION.md).

## Repository layout

```text
GeoEF-Aff_Zenodo_8_1_1/
  config.py                         central paths and hyperparameters
  main.py                           primary 80:10:10 training entry point
  model.py                          GeoEF-Aff model
  dynamic_modules.py                RAAD/geometric modules
  data_loader.py                    dataset and batching logic
  protein_features.py               structure features
  foldx_processor.py                FoldX interface
  precompute_samples.py             structure/FoldX preprocessing
  precompute_esm_embeddings.py      global and local ESM preprocessing
  eval_kfold_cv_calibrated_3feature.py
                                    optional three-fold evaluation
  models/best_model.pth             released final checkpoint
  splits/                            three frozen 80:10:10 partitions
  checksums.sha256                   integrity checks for critical artifacts
  uniair_external/                  HER2 and TCR--pMHC evaluation
  data/README.md                    expected data layout
  models/README.md                  checkpoint placement
  tests/                             release-level static checks
```

## Installation

Python 3.10 and a CUDA-capable Linux environment are recommended.

```bash
pip install -r requirements.txt
```

`torch-scatter` and `torch-cluster` must match the installed PyTorch and CUDA
versions. FoldX is separately licensed software and is not included.

## Data layout

Place the SKEMPI 2.0 table, structures, and precomputed features under `data/`
as described in [`data/README.md`](data/README.md). All locations can also be
overridden with environment variables:

```bash
export CSV_PATH=/path/to/skempi_v2.csv
export PDB_DIR=/path/to/PDBs
export PRECOMPUTED_DIR=/path/to/precomputed_samples
export FOLDX_PATH=/path/to/foldx
```

## Preprocessing

```bash
python precompute_samples.py --workers 4
python precompute_esm_embeddings.py
```

To reuse compatible structure/FoldX samples before adding local ESM tokens:

```bash
python prepare_localtoken_samples.py \
  --source-dir /path/to/source_precomputed_samples \
  --destination-dir /path/to/precomputed_samples
python precompute_esm_embeddings.py
```

## Primary 80:10:10 experiment

Single-process launch:

```bash
python main.py
```

Eight-GPU distributed launch:

```bash
bash run_8gpu.sh
```

The default run uses the frozen seed-42 partition from
`splits/fold_01/fold_01_split.json`. A newly trained model and runtime outputs
are written to:

```text
outputs/single_split_seed42/
  best_model.pth
  training.log
  training_audit.json
  validation_metrics.csv
  prediction_scatterplot.png
```

To train with another retained partition without overwriting the primary run:

```bash
RANDOM_SEED=43 \
SPLIT_JSON=splits/fold_02/fold_02_split.json \
OUTPUT_DIR=outputs/single_split_seed43 \
bash run_8gpu.sh
```

Inference and case-study scripts use `models/best_model.pth` by default. The
checkpoint file is not loaded or modified during split validation.

## Evaluation and case studies

```bash
python eval_pth_metrics.py --help
python run_case_study.py --help
python run_all_case_studies.py --help
python m595_blind_prediction_3feature.py --help
python uniair_external/run_pipeline.py --help
```

## Integrity verification

Verify the released checkpoint and split files:

```bash
sha256sum -c checksums.sha256
```
