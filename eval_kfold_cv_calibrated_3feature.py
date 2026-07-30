import argparse
import csv
import glob
import os
import sys
from collections import Counter, OrderedDict
from typing import Dict, List, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR
for module_path in (PROJECT_DIR, SCRIPT_DIR):
    if module_path in sys.path:
        sys.path.remove(module_path)
    sys.path.insert(0, module_path)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader, Dataset, Subset

import config
from esm_local_tokens import LOCAL_ESM_KEYS, LOCAL_TOKEN_VERSION_KEY
from data_loader import PrecomputedDataset, filter_none_collate
from model import ESM_FoldX_DDAffinity, ESM_RAAD_FoldX_DDAffinity
from train_utils import evaluate_model


class MutationCountFilteredDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optional three-fold evaluation entry point. Supports overall and "
            "mutation-count subset evaluation, with linear-regression-calibrated RMSE/MAE."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=getattr(config, "BEST_MODEL_PATH", os.path.join(SCRIPT_DIR, "best_model.pth")),
        help=(
            "Checkpoint path or glob pattern. Prefer passing one checkpoint per fold. "
            "If a single checkpoint is given, it will be reused for every fold and a warning will be shown."
        ),
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of group-based folds. Default: 3.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["overall", "mutation_count", "both"],
        default="both",
        help="Which evaluation scopes to run on each fold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffled KFold splitting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config.BATCH_SIZE for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override config.NUM_WORKERS for evaluation.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional CSV file to save per-fold and aggregate metrics.",
    )
    parser.add_argument(
        "--save-preds",
        type=str,
        default=None,
        help="Optional directory to save per-fold prediction CSV files.",
    )
    parser.add_argument(
        "--auroc-mode",
        choices=["sign", "threshold"],
        default="sign",
        help="How to binarize labels for AUROC. 'sign' follows DDAffinity: ddG > 0 is positive.",
    )
    parser.add_argument(
        "--auroc-threshold",
        type=float,
        default=0.0,
        help="Threshold used when auroc-mode=threshold.",
    )
    parser.add_argument(
        "--auroc-positive-if",
        choices=["greater", "less"],
        default="greater",
        help="Define positive class as labels greater than or less than the threshold.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["complex", "group", "random"],
        default=getattr(config, "SPLIT_MODE", "complex"),
        help="Use DDAffinity-like group K-fold by default, or random KFold for legacy comparison.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default=getattr(config, "SPLIT_GROUP_COL", "#Pdb"),
        help="CSV column used as the GroupKFold group. Default: #Pdb.",
    )
    return parser.parse_args()


def validate_runtime_config():
    feature_dim = getattr(config, "FOLDX_FEATURE_DIM", None)
    feature_mode = getattr(config, "FOLDX_FEATURE_MODE", None)
    if feature_dim != 3:
        raise RuntimeError(
            f"This 3-feature K-fold script requires config.FOLDX_FEATURE_DIM == 3, got {feature_dim!r}."
        )
    if feature_mode != "wt_mut_delta":
        raise RuntimeError(
            "This 3-feature K-fold script requires "
            f"config.FOLDX_FEATURE_MODE == 'wt_mut_delta', got {feature_mode!r}."
        )


def resolve_checkpoints(ckpt_arg):
    if ckpt_arg is None:
        default_ckpt = getattr(config, "BEST_MODEL_PATH", os.path.join(SCRIPT_DIR, "best_model.pth"))
        if os.path.exists(default_ckpt):
            return [default_ckpt]
        raise FileNotFoundError(
            "No checkpoint specified with --ckpt and default best_model.pth was not found."
        )

    if any(char in ckpt_arg for char in ["*", "?", "["]):
        patterns = [ckpt_arg]
        if not os.path.isabs(ckpt_arg):
            patterns.extend(
                [
                    os.path.join(SCRIPT_DIR, ckpt_arg),
                    os.path.join(config.BASE_DIR, ckpt_arg),
                ]
            )
        matched = []
        for pattern in patterns:
            matched.extend(glob.glob(pattern))
        matched = sorted(set(os.path.abspath(path) for path in matched))
        if not matched:
            raise FileNotFoundError(f"No checkpoint matched pattern: {ckpt_arg}")
        return matched

    if not os.path.isabs(ckpt_arg):
        candidates = [
            os.path.join(SCRIPT_DIR, ckpt_arg),
            os.path.join(config.BASE_DIR, ckpt_arg),
        ]
        ckpt_arg = next((path for path in candidates if os.path.exists(path)), candidates[-1])

    if not os.path.exists(ckpt_arg):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")

    return [ckpt_arg]


def assign_checkpoints_to_folds(checkpoints: Sequence[str], num_folds: int) -> List[str]:
    if len(checkpoints) == num_folds:
        return list(checkpoints)
    if len(checkpoints) == 1:
        print(
            "Warning: only one checkpoint was provided. It will be reused for all folds. "
            "This is not a full train-per-fold cross-validation setup."
        )
        return [checkpoints[0]] * num_folds
    raise ValueError(
        f"Expected either 1 checkpoint or exactly {num_folds} checkpoints, got {len(checkpoints)}."
    )


def build_dataset():
    precomputed_dir = getattr(
        config,
        "PRECOMPUTED_DIR",
        getattr(config, "PRECOMPUTED_DIR", os.path.join(SCRIPT_DIR, "precomputed_samples_3feature_esm650m")),
    )
    if not os.path.isdir(precomputed_dir):
        raise FileNotFoundError(
            f"3-feature precomputed sample directory not found: {precomputed_dir}. "
            "Run this experiment's precompute_samples.py first, or set PRECOMPUTED_DIR."
        )
    pt_files = [name for name in os.listdir(precomputed_dir) if name.endswith(".pt")]
    if not pt_files:
        raise RuntimeError(
            f"No .pt files found in 3-feature precomputed directory: {precomputed_dir}. "
            "Run this experiment's precompute_samples.py first, or set PRECOMPUTED_DIR."
        )

    print(f"Using 3-feature precomputed dataset from {precomputed_dir}")
    dataset = PrecomputedDataset(precomputed_dir)
    if len(dataset) == 0:
        raise RuntimeError(
            "No valid 3-feature precomputed samples match the current FoldX/ESM settings. "
            "Run this experiment's precompute_samples.py and precompute_esm_embeddings.py, "
            "or set PRECOMPUTED_DIR"
        )
    return dataset


def dataset_has_precomputed_esm(dataset, max_checks: int = 5):
    checked = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue
        checked += 1
        has_embeddings = (
            "wt_esm_embedding" in sample
            and "mut_esm_embedding" in sample
            and "mutation_esm_embedding" in sample
            and sample["wt_esm_embedding"] is not None
            and sample["mut_esm_embedding"] is not None
            and sample["mutation_esm_embedding"] is not None
            and LOCAL_ESM_KEYS.issubset(sample)
            and sample.get(LOCAL_TOKEN_VERSION_KEY)
            == getattr(config, "ESM_LOCAL_TOKEN_VERSION", 1)
            and sample.get("esm_mutation_window_radius") == getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8)
        )
        if not has_embeddings:
            print(
                f"Sample {idx} is missing current precomputed ESM embeddings. "
                "Falling back to runtime ESM inference."
            )
            return False
        if checked >= max_checks:
            return True

    print("Could not confirm precomputed ESM embeddings from dataset samples.")
    return False


def build_dataloader(eval_dataset, batch_size, num_workers):
    kwargs = {
        "dataset": eval_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": filter_none_collate,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True

    return DataLoader(**kwargs)


def build_model(use_precomputed_esm=None):
    if use_precomputed_esm is None:
        use_precomputed_esm = getattr(config, "USE_PRECOMPUTED_ESM", False)

    if config.USE_DYNAMIC_MODELING:
        return ESM_RAAD_FoldX_DDAffinity(
            esm_model_name=config.ESM_MODEL_NAME,
            hidden_dim=config.HIDDEN_DIM,
            raad_hidden_dim=config.RAAD_HIDDEN_DIM,
            raad_layers=config.RAAD_LAYERS,
            dropout=config.DROPOUT,
            edge_types=config.EDGE_TYPES,
            rball_radius=config.RBALL_RADIUS,
            knn_k=config.KNN_K,
            use_atom_features=config.USE_ATOM_FEATURES,
            use_precomputed_esm=use_precomputed_esm,
            local_radius=getattr(config, "MUTATION_LOCAL_RADIUS", 10.0),
            esm_mutation_window_radius=getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8),
            esm_local_max_tokens=getattr(config, "ESM_LOCAL_MAX_TOKENS", 32),
            struct_local_max_residues=getattr(config, "STRUCT_LOCAL_MAX_RESIDUES", 32),
            coords_agg=getattr(config, "COORDS_AGG", "mean"),
        )

    return ESM_FoldX_DDAffinity(
        esm_model_name=config.ESM_MODEL_NAME,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        use_precomputed_esm=use_precomputed_esm,
    )


def load_weights(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {ckpt_path}")

    load_errors = []
    try:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint with strict key matching.")
    except RuntimeError:
        load_errors.append("raw state_dict failed")
        stripped_state = OrderedDict()
        prefixed_state = OrderedDict()

        for key, value in checkpoint.items():
            stripped_state[key.replace("module.", "", 1)] = value
            prefixed_state[key if key.startswith("module.") else f"module.{key}"] = value

        try:
            model.load_state_dict(stripped_state)
            print("Loaded checkpoint after stripping 'module.' prefixes.")
        except RuntimeError as stripped_error:
            load_errors.append(f"stripped state_dict failed: {stripped_error}")
            try:
                model.load_state_dict(prefixed_state)
                print("Loaded checkpoint after adding 'module.' prefixes.")
            except RuntimeError as prefixed_error:
                load_errors.append(f"prefixed state_dict failed: {prefixed_error}")
                raise RuntimeError(
                    "Failed to strictly load checkpoint into the 3-feature K-fold model. "
                    "Use a checkpoint trained by this localtoken32 experiment; "
                    "older compressed-token or 4D checkpoints are incompatible. "
                    f"Load attempts: {' | '.join(load_errors)}"
                ) from prefixed_error

    model.to(device)
    model.eval()
    return model


def compute_auroc(labels, preds, mode, threshold, positive_if):
    if mode == "sign":
        binary_labels = (labels > 0).astype(int)
    else:
        if positive_if == "greater":
            binary_labels = (labels > threshold).astype(int)
        else:
            binary_labels = (labels < threshold).astype(int)

    if len(np.unique(binary_labels)) < 2:
        return np.nan

    return roc_auc_score(binary_labels, preds)


def compute_metrics_with_linear_calibration(
    labels,
    preds,
    avg_loss,
    auroc_mode,
    auroc_threshold,
    auroc_positive_if,
):
    labels = np.asarray(labels, dtype=float)
    preds = np.asarray(preds, dtype=float)

    if len(labels) < 2:
        return {
            "loss_mse": avg_loss,
            "pearson": np.nan,
            "spearman": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "auroc": np.nan,
            "samples": len(labels),
            "calibration_slope": np.nan,
            "calibration_intercept": np.nan,
        }

    pearson_corr, _ = pearsonr(labels, preds)
    spearman_corr, _ = spearmanr(labels, preds)

    slope, intercept = np.polyfit(preds, labels, deg=1)
    calibrated_preds = slope * preds + intercept

    mse = np.mean((labels - calibrated_preds) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(labels - calibrated_preds))
    auroc = compute_auroc(labels, preds, auroc_mode, auroc_threshold, auroc_positive_if)

    return {
        "loss_mse": avg_loss,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "rmse": rmse,
        "mae": mae,
        "auroc": auroc,
        "samples": len(labels),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def save_predictions_csv(output_dir, ckpt_path, labels, preds, fold_index: int, scope_name: str):
    os.makedirs(output_dir, exist_ok=True)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_path = os.path.join(output_dir, f"{ckpt_name}_fold{fold_index}_{scope_name}_predictions.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prediction"])
        for idx, (label, pred) in enumerate(zip(labels, preds)):
            writer.writerow([idx, float(label), float(pred)])

    return out_path


def read_training_dataframe() -> pd.DataFrame:
    required = {"#Pdb", "Mutation(s)_cleaned", "Affinity_mut_parsed", "Affinity_wt_parsed"}
    attempts = [
        {"sep": ";"},
        {"sep": None, "engine": "python"},
    ]
    errors = []
    for kwargs in attempts:
        try:
            df = pd.read_csv(config.CSV_PATH, **kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
            continue
        if required.issubset(df.columns):
            break
        errors.append(f"{kwargs}: missing columns {sorted(required.difference(df.columns))}")
    else:
        raise RuntimeError(
            f"Could not read a SKEMPI-style CSV from config.CSV_PATH={config.CSV_PATH!r}. "
            f"Attempts: {' | '.join(errors)}"
        )

    df = df.dropna(
        subset=["#Pdb", "Mutation(s)_cleaned", "Affinity_mut_parsed", "Affinity_wt_parsed"]
    ).reset_index(drop=True)
    df["ddG"] = (
        (8.314 / 4184)
        * (273.15 + 25.0)
        * (
            np.log(df["Affinity_mut_parsed"].astype(float))
            - np.log(df["Affinity_wt_parsed"].astype(float))
        )
    )
    return df


def get_base_dataframe(dataset: Dataset):
    if hasattr(dataset, "df"):
        return dataset.df
    if isinstance(dataset, PrecomputedDataset):
        df = read_training_dataframe()
        if dataset.indices and max(dataset.indices) >= len(df):
            raise IndexError(
                f"Precomputed sample index {max(dataset.indices)} exceeds CSV rows {len(df)}. "
                "Regenerate precomputed samples from the same CSV_PATH used for evaluation."
            )
        return df
    return None


def _group_value(row: pd.Series, group_col: str) -> str:
    if group_col == "pdb_id":
        return str(row["#Pdb"]).split("_")[0]
    if group_col not in row or pd.isna(row[group_col]) or str(row[group_col]).strip() == "":
        return str(row["#Pdb"])
    return str(row[group_col])


def get_dataset_rows(dataset: Dataset, group_col: str) -> pd.DataFrame:
    if isinstance(dataset, PrecomputedDataset):
        df = get_base_dataframe(dataset)
        rows = df.iloc[dataset.indices].copy().reset_index(drop=True)
    elif hasattr(dataset, "df"):
        rows = dataset.df.reset_index(drop=True).copy()
    else:
        raise TypeError("Group split requires PrecomputedDataset or a dataset with a .df attribute.")

    rows["_split_group"] = rows.apply(lambda row: _group_value(row, group_col), axis=1)
    return rows


def group_count_summary(groups: List[str]) -> str:
    counts = Counter(groups)
    values = np.array(list(counts.values()), dtype=float)
    return (
        f"groups={len(counts)}, min={int(values.min())}, "
        f"median={float(np.median(values)):.1f}, max={int(values.max())}"
    )


def make_group_kfold_splits(dataset: Dataset, n_splits: int, group_col: str):
    rows = get_dataset_rows(dataset, group_col)
    groups = rows["_split_group"].astype(str).tolist()
    unique_groups = len(set(groups))
    if unique_groups < n_splits:
        raise ValueError(f"Need at least {n_splits} groups for GroupKFold, got {unique_groups}.")
    indices = np.arange(len(dataset))
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(indices, groups=groups)), groups


def count_mutations_from_string(mutation_str: str) -> int:
    return len([x.strip() for x in str(mutation_str).split(",") if x.strip()])


def build_mutation_subsets(dataset: Dataset, eval_indices: List[int]) -> Dict[str, List[int]]:
    df = get_base_dataframe(dataset)
    subsets = {"all": list(eval_indices), "single": [], "multiple": []}

    if df is None:
        raise ValueError(
            "Mutation-count subset evaluation currently requires a dataset with a .df DataFrame, "
            "such as SKEMPI2Dataset. PrecomputedDataset does not retain mutation annotations."
        )

    for dataset_idx in eval_indices:
        if isinstance(dataset, PrecomputedDataset):
            csv_idx = dataset.indices[dataset_idx]
            row = df.iloc[csv_idx]
        else:
            row = df.iloc[dataset_idx]
        mutation_str = row["Mutation(s)_cleaned"]
        n_mut = count_mutations_from_string(mutation_str)
        if n_mut == 1:
            subsets["single"].append(dataset_idx)
        elif n_mut >= 2:
            subsets["multiple"].append(dataset_idx)

    return subsets


def evaluate_once(model, dataloader, criterion, device, args):
    avg_loss, _, labels, preds = evaluate_model(model, dataloader, criterion, device)
    metrics = compute_metrics_with_linear_calibration(
        labels,
        preds,
        avg_loss,
        args.auroc_mode,
        args.auroc_threshold,
        args.auroc_positive_if,
    )
    return metrics, labels, preds


def print_row(row: Dict[str, object]):
    scope = row.get("subset") or row.get("eval_scope", "overall")
    fold_text = row.get("fold", "summary")
    print(f"Fold      : {fold_text}")
    print(f"Scope     : {scope}")
    print(f"Samples   : {row['samples']}")
    print(f"Loss(MSE) : {row['loss_mse']:.6f}")
    print(f"Pearson   : {row['pearson']:.6f}")
    print(f"Spearman  : {row['spearman']:.6f}")
    print(f"RMSE(cal) : {row['rmse']:.6f}")
    print(f"MAE(cal)  : {row['mae']:.6f}")
    print(f"AUROC     : {row['auroc']:.6f}")
    print(f"Slope     : {row['calibration_slope']:.6f}")
    print(f"Intercept : {row['calibration_intercept']:.6f}")


def save_results_csv(csv_path: str, rows: Sequence[Dict[str, object]]):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    numeric_keys = [
        "samples",
        "loss_mse",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "auroc",
        "calibration_slope",
        "calibration_intercept",
    ]

    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        scope = row.get("eval_scope", "")
        subset = row.get("subset", "")
        grouped.setdefault((scope, subset), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (scope, subset), group_rows in grouped.items():
        summary = {
            "checkpoint": "aggregate",
            "fold": "mean_std",
            "eval_scope": scope,
            "subset": subset,
            "num_folds_aggregated": len(group_rows),
        }
        for key in numeric_keys:
            values = np.array([row[key] for row in group_rows], dtype=float)
            summary[f"{key}_mean"] = float(np.nanmean(values))
            summary[f"{key}_std"] = float(np.nanstd(values))
        summary_rows.append(summary)

    return summary_rows


def evaluate_overall_fold(
    ckpt_path: str,
    dataset,
    eval_indices: List[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    args,
    use_precomputed_esm: bool,
):
    eval_dataset = MutationCountFilteredDataset(dataset, eval_indices)
    dataloader = build_dataloader(eval_dataset, batch_size, num_workers)
    model = build_model(use_precomputed_esm=use_precomputed_esm)
    model = load_weights(model, ckpt_path, device)
    criterion = nn.MSELoss()
    metrics, labels, preds = evaluate_once(model, dataloader, criterion, device, args)
    row = {
        "checkpoint": ckpt_path,
        "eval_scope": "overall",
        **metrics,
    }
    return row, labels, preds


def evaluate_mutation_count_fold(
    ckpt_path: str,
    dataset: Dataset,
    eval_indices: List[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    args,
    use_precomputed_esm: bool,
):
    subsets = build_mutation_subsets(dataset, eval_indices)
    criterion = nn.MSELoss()
    rows = []
    preds_payload = []

    for subset_name in ["all", "single", "multiple"]:
        subset_indices = subsets[subset_name]
        if len(subset_indices) == 0:
            rows.append(
                {
                    "checkpoint": ckpt_path,
                    "eval_scope": "mutation_count",
                    "subset": subset_name,
                    "samples": 0,
                    "loss_mse": np.nan,
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "auroc": np.nan,
                    "calibration_slope": np.nan,
                    "calibration_intercept": np.nan,
                }
            )
            preds_payload.append((subset_name, np.array([]), np.array([])))
            continue

        subset_dataset = MutationCountFilteredDataset(dataset, subset_indices)
        dataloader = build_dataloader(subset_dataset, batch_size, num_workers)
        model = build_model(use_precomputed_esm=use_precomputed_esm)
        model = load_weights(model, ckpt_path, device)
        metrics, labels, preds = evaluate_once(model, dataloader, criterion, device, args)
        rows.append(
            {
                "checkpoint": ckpt_path,
                "eval_scope": "mutation_count",
                "subset": subset_name,
                **metrics,
            }
        )
        preds_payload.append((subset_name, labels, preds))

    return rows, preds_payload


def main():
    args = parse_args()
    validate_runtime_config()

    batch_size = args.batch_size if args.batch_size is not None else config.BATCH_SIZE
    num_workers = args.num_workers if args.num_workers is not None else config.NUM_WORKERS
    device = config.DEVICE
    print("3-feature K-fold evaluation")
    print(f"FoldX feature mode: {config.FOLDX_FEATURE_MODE} ([wt_energy, mut_energy, mut_energy - wt_energy])")
    print(f"CSV_PATH: {config.CSV_PATH}")
    print(f"PRECOMPUTED_DIR: {config.PRECOMPUTED_DIR}")

    checkpoints = resolve_checkpoints(args.ckpt)
    fold_checkpoints = assign_checkpoints_to_folds(checkpoints, args.num_folds)
    dataset = build_dataset()
    use_precomputed_esm = getattr(config, "USE_PRECOMPUTED_ESM", False)
    if use_precomputed_esm:
        use_precomputed_esm = dataset_has_precomputed_esm(dataset)
        print(f"USE_PRECOMPUTED_ESM resolved to: {use_precomputed_esm}")

    if args.split_mode in {"complex", "group"}:
        fold_splits, groups = make_group_kfold_splits(dataset, args.num_folds, group_col=args.group_col)
        print(
            "Using DDAffinity-like GroupKFold: "
            f"group_col={args.group_col}, {group_count_summary(groups)}"
        )
    else:
        kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(kfold.split(range(len(dataset))))
        print("Using legacy random KFold. This is not complex-level evaluation.")

    all_rows: List[Dict[str, object]] = []

    for fold_index, (_, test_indices) in enumerate(fold_splits, start=1):
        ckpt_path = fold_checkpoints[fold_index - 1]
        eval_indices = list(map(int, test_indices))

        print("=" * 80)
        print(f"Fold {fold_index}/{args.num_folds}")
        print(f"Evaluating checkpoint: {ckpt_path}")
        print(f"Fold size: {len(eval_indices)}")

        if args.eval_mode in {"overall", "both"}:
            print("-" * 80)
            overall_row, labels, preds = evaluate_overall_fold(
                ckpt_path,
                dataset,
                eval_indices,
                batch_size,
                num_workers,
                device,
                args,
                use_precomputed_esm,
            )
            overall_row["fold"] = fold_index
            overall_row["seed"] = args.seed
            print_row(overall_row)
            all_rows.append(overall_row)
            if args.save_preds:
                pred_path = save_predictions_csv(args.save_preds, ckpt_path, labels, preds, fold_index, "overall")
                print(f"Saved predictions to: {pred_path}")

        if args.eval_mode in {"mutation_count", "both"}:
            rows, preds_payload = evaluate_mutation_count_fold(
                ckpt_path,
                dataset,
                eval_indices,
                batch_size,
                num_workers,
                device,
                args,
                use_precomputed_esm,
            )
            for row in rows:
                print("-" * 80)
                row["fold"] = fold_index
                row["seed"] = args.seed
                print_row(row)
                all_rows.append(row)
            if args.save_preds:
                for subset_name, labels, preds in preds_payload:
                    pred_path = save_predictions_csv(
                        args.save_preds,
                        ckpt_path,
                        labels,
                        preds,
                        fold_index,
                        subset_name,
                    )
                    print(f"Saved predictions to: {pred_path}")

    summary_rows = aggregate_rows(all_rows)
    for row in summary_rows:
        print("=" * 80)
        scope = row.get("subset") or row.get("eval_scope", "overall")
        print(f"Aggregate Scope : {scope}")
        print(f"Folds           : {row['num_folds_aggregated']}")
        print(f"Pearson mean/std: {row['pearson_mean']:.6f} / {row['pearson_std']:.6f}")
        print(f"Spearman mean/std: {row['spearman_mean']:.6f} / {row['spearman_std']:.6f}")
        print(f"RMSE mean/std   : {row['rmse_mean']:.6f} / {row['rmse_std']:.6f}")
        print(f"MAE mean/std    : {row['mae_mean']:.6f} / {row['mae_std']:.6f}")
        print(f"AUROC mean/std  : {row['auroc_mean']:.6f} / {row['auroc_std']:.6f}")

    if args.csv_out:
        save_results_csv(args.csv_out, list(all_rows) + list(summary_rows))
        print("=" * 80)
        print(f"Saved summary metrics to: {args.csv_out}")


if __name__ == "__main__":
    main()
