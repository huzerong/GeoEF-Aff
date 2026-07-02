#!/usr/bin/env python3
"""Zero-shot three-fold evaluation using only ESM-2 150M cached embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

import config
from cv_split_utils import assign_pdb_group_folds, assign_sample_folds
from data_loader import PrecomputedDataset


ESM_MODEL_NAME = "esm2_t30_150M_UR50D"
ESM_DIM = 640
LOCAL_DIM = ESM_DIM * 4
MODEL_ID = "esm2_t30_150M_UR50D_zero_shot"
DEFAULT_PRIMARY_SCORE = "local_combined_l2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate zero-shot ESM-2 150M embedding perturbation scores in "
            "three folds. No training, optimizer, or checkpoint is used."
        )
    )
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=config.CV_RANDOM_SEED)
    parser.add_argument(
        "--split-mode",
        choices=("sample", "pdb_group"),
        default="sample",
        help="sample is the requested default; pdb_group is available for comparison.",
    )
    parser.add_argument(
        "--precomputed-dir",
        type=Path,
        default=Path(config.PRECOMPUTED_DIR),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.VARIANT_DIR) / "three_fold_results_esm2_150m_zeroshot",
    )
    parser.add_argument(
        "--skempi-csv",
        type=Path,
        default=Path(config.CSV_PATH),
        help=(
            "SKEMPI2 CSV used only to recover Mutation(s)_cleaned for "
            "single/multiple subgroup metrics."
        ),
    )
    parser.add_argument(
        "--primary-score",
        choices=(
            "global_l2",
            "global_l1",
            "global_cosine_distance",
            "site_delta_l2",
            "window_delta_l2",
            "local_combined_l2",
        ),
        default=DEFAULT_PRIMARY_SCORE,
        help="Score copied to pred_ddg in prediction CSVs.",
    )
    parser.add_argument(
        "--orientation",
        choices=("positive", "negative"),
        default="positive",
        help=(
            "positive means larger embedding perturbation predicts larger ddG. "
            "negative flips the sign. Both orientations are still reported in score_metrics.csv."
        ),
    )
    parser.add_argument(
        "--auroc-threshold",
        type=float,
        default=0.0,
        help="Label threshold for AUROC. Default follows DDAffinity: ddG > 0 is positive.",
    )
    parser.add_argument(
        "--auroc-positive-if",
        choices=("greater", "less"),
        default="greater",
        help="Use labels greater or less than --auroc-threshold as AUROC positives.",
    )
    return parser.parse_args()


def scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def as_vector(sample: Dict[str, object], key: str, expected_dim: int, position: int) -> torch.Tensor:
    if key not in sample:
        raise KeyError(f"Sample at dataset position {position} is missing {key}.")
    vector = torch.as_tensor(sample[key], dtype=torch.float32).view(-1)
    if int(vector.numel()) != expected_dim:
        raise ValueError(
            f"Sample at dataset position {position} has bad {key} dim: "
            f"numel={vector.numel()}, expected={expected_dim}."
        )
    return vector


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.norm(left) * torch.linalg.norm(right)
    if float(denominator.item()) == 0.0:
        return 0.0
    cosine = torch.dot(left, right) / denominator
    return float((1.0 - cosine).item())


def compute_scores(sample: Dict[str, object], position: int) -> Dict[str, float]:
    wt = as_vector(sample, "wt_esm_embedding", ESM_DIM, position)
    mut = as_vector(sample, "mut_esm_embedding", ESM_DIM, position)
    local = as_vector(sample, "mutation_esm_embedding", LOCAL_DIM, position)

    delta = mut - wt
    site_delta = local[ESM_DIM * 2 : ESM_DIM * 3]
    window_delta = local[ESM_DIM * 3 : ESM_DIM * 4]
    site_l2 = torch.linalg.norm(site_delta)
    window_l2 = torch.linalg.norm(window_delta)
    return {
        "global_l2": float(torch.linalg.norm(delta).item()),
        "global_l1": float(torch.mean(torch.abs(delta)).item()),
        "global_cosine_distance": cosine_distance(wt, mut),
        "site_delta_l2": float(site_l2.item()),
        "window_delta_l2": float(window_l2.item()),
        "local_combined_l2": float(torch.sqrt(site_l2.square() + window_l2.square()).item()),
    }


def safe_corr(func, labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) < 2 or np.std(labels) == 0.0 or np.std(predictions) == 0.0:
        return 0.0
    value = float(func(labels, predictions)[0])
    return 0.0 if np.isnan(value) else value


def compute_auroc(
    labels: Sequence[float],
    predictions: Sequence[float],
    threshold: float,
    positive_if: str,
) -> float:
    labels_array = np.asarray(labels, dtype=float)
    predictions_array = np.asarray(predictions, dtype=float)
    if positive_if == "greater":
        binary_labels = (labels_array > threshold).astype(int)
    else:
        binary_labels = (labels_array < threshold).astype(int)

    positives = int(binary_labels.sum())
    negatives = int(len(binary_labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = rankdata(predictions_array, method="average")
    positive_rank_sum = float(ranks[binary_labels == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)


def regression_metrics(
    labels: Sequence[float],
    predictions: Sequence[float],
    auroc_threshold: float,
    auroc_positive_if: str,
) -> Dict[str, float]:
    labels_array = np.asarray(labels, dtype=float)
    predictions_array = np.asarray(predictions, dtype=float)
    residual = predictions_array - labels_array
    return {
        "samples": int(len(labels_array)),
        "pearson": safe_corr(pearsonr, labels_array, predictions_array),
        "spearman": safe_corr(spearmanr, labels_array, predictions_array),
        "rmse": float(np.sqrt(np.mean(residual**2))) if len(residual) else float("nan"),
        "mae": float(np.mean(np.abs(residual))) if len(residual) else float("nan"),
        "auroc": compute_auroc(
            labels_array,
            predictions_array,
            auroc_threshold,
            auroc_positive_if,
        )
        if len(labels_array)
        else float("nan"),
    }


def count_mutations_from_string(mutation_str: str) -> int:
    return len([item.strip() for item in str(mutation_str).split(",") if item.strip()])


def mutation_type_from_count(count: int) -> str:
    if count == 1:
        return "single"
    if count >= 2:
        return "multiple"
    return "unknown"


def load_skempi_rows(path: Path) -> List[Dict[str, str]]:
    required = ["#Pdb", "Mutation(s)_cleaned", "Affinity_mut_parsed", "Affinity_wt_parsed"]
    rows = []
    with path.expanduser().open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        missing = [column for column in required if column not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"SKEMPI2 CSV is missing required columns: {missing}")
        for row in reader:
            keep = True
            for column in required:
                value = str(row.get(column, "")).strip()
                if value == "" or value.lower() in {"nan", "none"}:
                    keep = False
                    break
            if keep:
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No usable SKEMPI2 rows found in {path}.")
    return rows


def build_splits(
    rows: Sequence[Dict[str, object]], num_folds: int, seed: int, split_mode: str
):
    group_ids = [str(row["pdb_id"]) for row in rows]
    if split_mode == "sample":
        folds = assign_sample_folds(len(rows), num_folds, seed)
        fold_groups = [
            sorted({group_ids[position] for position in positions})
            for positions in folds
        ]
    elif split_mode == "pdb_group":
        folds, fold_groups = assign_pdb_group_folds(group_ids, num_folds, seed)
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")
    return folds, fold_groups


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def signed_score(value: float, orientation: str) -> float:
    return value if orientation == "positive" else -value


def load_rows(
    dataset: PrecomputedDataset,
    skempi_rows: Sequence[Dict[str, str]],
    primary_score: str,
    orientation: str,
) -> List[Dict[str, object]]:
    rows = []
    for position in range(len(dataset)):
        sample = dataset[position]
        scores = compute_scores(sample, position)
        pdb_id = str(sample.get("pdb_id", "")).strip() or "UNKNOWN"
        sample_index = int(dataset.indices[position])
        if sample_index >= len(skempi_rows):
            raise IndexError(
                f"Precomputed sample index {sample_index} is outside the "
                f"filtered SKEMPI2 metadata length {len(skempi_rows)}."
            )
        mutation_string = str(skempi_rows[sample_index]["Mutation(s)_cleaned"])
        mutation_count = count_mutations_from_string(mutation_string)
        row = {
            "dataset_position": position,
            "sample_index": sample_index,
            "pdb_id": pdb_id,
            "mutation_string": mutation_string,
            "mutation_count": mutation_count,
            "mutation_type": mutation_type_from_count(mutation_count),
            "label_ddg": scalar(sample["delta_g"]),
            "primary_score": primary_score,
            "orientation": orientation,
            "pred_ddg": signed_score(scores[primary_score], orientation),
            **scores,
        }
        rows.append(row)
    return rows


def summarize_scope(
    rows: Sequence[Dict[str, object]],
    scope: str,
    score_names: Sequence[str],
    auroc_threshold: float,
    auroc_positive_if: str,
) -> List[Dict[str, object]]:
    metrics_rows = []
    for mutation_type, subset_rows in mutation_subsets(rows).items():
        labels = [float(row["label_ddg"]) for row in subset_rows]
        for score_name in score_names:
            raw_scores = [float(row[score_name]) for row in subset_rows]
            for orientation, multiplier in (("positive", 1.0), ("negative", -1.0)):
                predictions = [multiplier * score for score in raw_scores]
                metrics_rows.append(
                    {
                        "scope": scope,
                        "mutation_type": mutation_type,
                        "score": score_name,
                        "orientation": orientation,
                        **regression_metrics(
                            labels,
                            predictions,
                            auroc_threshold,
                            auroc_positive_if,
                        ),
                    }
                )
    return metrics_rows


def mutation_subsets(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    return {
        "all": list(rows),
        "single": [row for row in rows if row.get("mutation_type") == "single"],
        "multiple": [row for row in rows if row.get("mutation_type") == "multiple"],
    }


def summarize_primary_score(
    rows: Sequence[Dict[str, object]],
    scope: str,
    auroc_threshold: float,
    auroc_positive_if: str,
) -> List[Dict[str, object]]:
    summary_rows = []
    for mutation_type, subset_rows in mutation_subsets(rows).items():
        labels = [float(row["label_ddg"]) for row in subset_rows]
        predictions = [float(row["pred_ddg"]) for row in subset_rows]
        summary_rows.append(
            {
                "scope": scope,
                "mutation_type": mutation_type,
                **regression_metrics(
                    labels,
                    predictions,
                    auroc_threshold,
                    auroc_positive_if,
                ),
            }
        )
    return summary_rows


def main() -> None:
    args = parse_args()
    if args.num_folds != 3:
        raise ValueError("This zero-shot experiment is designed for exactly three folds.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PrecomputedDataset(
        str(args.precomputed_dir.expanduser().resolve()),
        validate_current_config=False,
    )
    skempi_rows = load_skempi_rows(args.skempi_csv)
    rows = load_rows(dataset, skempi_rows, args.primary_score, args.orientation)
    folds, fold_groups = build_splits(rows, args.num_folds, args.seed, args.split_mode)

    for fold_index, positions in enumerate(folds):
        fold_number = fold_index + 1
        for position in positions:
            rows[position]["fold"] = fold_number

    fold_assignment_rows = [
        {
            "dataset_position": row["dataset_position"],
            "sample_index": row["sample_index"],
            "pdb_id": row["pdb_id"],
            "fold": row["fold"],
        }
        for row in rows
    ]
    write_csv(
        output_dir / "fold_assignments.csv",
        ["dataset_position", "sample_index", "pdb_id", "fold"],
        fold_assignment_rows,
    )

    split_summary = {
        "model": MODEL_ID,
        "esm_model_name": ESM_MODEL_NAME,
        "num_folds": args.num_folds,
        "seed": args.seed,
        "split_mode": args.split_mode,
        "uses_training": False,
        "uses_foldx": False,
        "uses_structure": False,
        "primary_score": args.primary_score,
        "orientation": args.orientation,
        "auroc_threshold": args.auroc_threshold,
        "auroc_positive_if": args.auroc_positive_if,
        "folds": [
            {
                "fold": index + 1,
                "samples": len(folds[index]),
                "unique_pdbs": len(fold_groups[index]),
            }
            for index in range(args.num_folds)
        ],
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )

    prediction_fieldnames = [
        "fold",
        "dataset_position",
        "sample_index",
        "pdb_id",
        "mutation_string",
        "mutation_count",
        "mutation_type",
        "label_ddg",
        "pred_ddg",
        "primary_score",
        "orientation",
        "global_l2",
        "global_l1",
        "global_cosine_distance",
        "site_delta_l2",
        "window_delta_l2",
        "local_combined_l2",
    ]
    oof_rows = []
    fold_summary_rows = []
    score_metric_rows = []
    score_names = [
        "global_l2",
        "global_l1",
        "global_cosine_distance",
        "site_delta_l2",
        "window_delta_l2",
        "local_combined_l2",
    ]

    for fold_index, positions in enumerate(folds):
        fold_number = fold_index + 1
        fold_dir = output_dir / f"fold_{fold_number}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_rows = [rows[position] for position in positions]
        write_csv(fold_dir / "validation_predictions.csv", prediction_fieldnames, fold_rows)
        oof_rows.extend(fold_rows)

        primary_summary_rows = summarize_primary_score(
            fold_rows,
            f"fold_{fold_number}",
            args.auroc_threshold,
            args.auroc_positive_if,
        )
        primary_metrics_all = next(
            row for row in primary_summary_rows if row["mutation_type"] == "all"
        )
        train_pdbs = set()
        for index in range(args.num_folds):
            if index != fold_index:
                train_pdbs.update(fold_groups[index])
        val_pdbs = set(fold_groups[fold_index])
        complete = {
            "model": MODEL_ID,
            "esm_model_name": ESM_MODEL_NAME,
            "uses_training": False,
            "uses_foldx": False,
            "uses_structure": False,
            "fold": fold_number,
            "split_mode": args.split_mode,
            "split_seed": args.seed,
            "primary_score": args.primary_score,
            "orientation": args.orientation,
            "train_samples": len(rows) - len(fold_rows),
            "validation_samples": len(fold_rows),
            "train_unique_pdbs": len(train_pdbs),
            "validation_unique_pdbs": len(val_pdbs),
            "shared_pdbs": len(train_pdbs.intersection(val_pdbs)),
            **{
                key: value
                for key, value in primary_metrics_all.items()
                if key not in {"scope", "mutation_type"}
            },
        }
        (fold_dir / "fold_complete.json").write_text(
            json.dumps(complete, indent=2), encoding="utf-8"
        )
        for primary_row in primary_summary_rows:
            fold_summary_rows.append(
                {
                    **primary_row,
                    "model": MODEL_ID,
                    "esm_model_name": ESM_MODEL_NAME,
                    "uses_training": False,
                    "uses_foldx": False,
                    "uses_structure": False,
                    "split_mode": args.split_mode,
                    "split_seed": args.seed,
                    "primary_score": args.primary_score,
                    "orientation": args.orientation,
                    "auroc_threshold": args.auroc_threshold,
                    "auroc_positive_if": args.auroc_positive_if,
                    "train_samples": len(rows) - len(fold_rows),
                    "validation_samples": len(fold_rows),
                    "train_unique_pdbs": len(train_pdbs),
                    "validation_unique_pdbs": len(val_pdbs),
                    "shared_pdbs": len(train_pdbs.intersection(val_pdbs)),
                }
            )
        score_metric_rows.extend(
            summarize_scope(
                fold_rows,
                f"fold_{fold_number}",
                score_names,
                args.auroc_threshold,
                args.auroc_positive_if,
            )
        )

    oof_rows.sort(key=lambda row: int(row["dataset_position"]))
    write_csv(output_dir / "oof_predictions.csv", prediction_fieldnames, oof_rows)

    oof_primary_rows = summarize_primary_score(
        oof_rows,
        "OOF",
        args.auroc_threshold,
        args.auroc_positive_if,
    )
    for oof_primary_row in oof_primary_rows:
        fold_summary_rows.append(
            {
                **oof_primary_row,
                "model": MODEL_ID,
                "esm_model_name": ESM_MODEL_NAME,
                "uses_training": False,
                "uses_foldx": False,
                "uses_structure": False,
                "split_mode": args.split_mode,
                "split_seed": args.seed,
                "primary_score": args.primary_score,
                "orientation": args.orientation,
                "auroc_threshold": args.auroc_threshold,
                "auroc_positive_if": args.auroc_positive_if,
            }
        )
    score_metric_rows.extend(
        summarize_scope(
            oof_rows,
            "OOF",
            score_names,
            args.auroc_threshold,
            args.auroc_positive_if,
        )
    )

    summary_fieldnames = []
    for row in fold_summary_rows:
        for key in row:
            if key not in summary_fieldnames:
                summary_fieldnames.append(key)
    write_csv(output_dir / "cv_summary.csv", summary_fieldnames, fold_summary_rows)

    score_metric_fieldnames = [
        "scope",
        "mutation_type",
        "score",
        "orientation",
        "samples",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "auroc",
    ]
    write_csv(output_dir / "score_metrics.csv", score_metric_fieldnames, score_metric_rows)

    skempi2_subgroup_rows = [
        row for row in fold_summary_rows if row["scope"] == "OOF"
    ]
    write_csv(
        output_dir / "skempi2_subgroup_metrics.csv",
        [
            "scope",
            "mutation_type",
            "samples",
            "pearson",
            "spearman",
            "rmse",
            "mae",
            "auroc",
            "primary_score",
            "orientation",
            "auroc_threshold",
            "auroc_positive_if",
        ],
        skempi2_subgroup_rows,
    )

    summary_json = {
        **split_summary,
        "oof_metrics_primary_by_mutation_type": oof_primary_rows,
        "note": (
            "RMSE/MAE are unscaled because zero-shot ESM perturbation scores are "
            "not calibrated kcal/mol ddG predictions."
        ),
    }
    (output_dir / "cv_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    print(f"Zero-shot ESM2-150M three-fold evaluation complete: {output_dir}")
    print(f"Primary summary: {output_dir / 'cv_summary.csv'}")
    print(f"All score metrics: {output_dir / 'score_metrics.csv'}")


if __name__ == "__main__":
    main()
