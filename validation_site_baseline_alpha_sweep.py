from __future__ import annotations

import argparse
import csv
import json
import math
import os
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ALPHAS = tuple(index / 10.0 for index in range(11))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer raw and self-mutation predictions on the training validation "
            "split, then select a site-baseline correction alpha."
        )
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint path. Default: best_model.pth.",
    )
    parser.add_argument(
        "--split-json",
        default=None,
        help="Default: splits/complex_split_80_10_10_seed<RANDOM_SEED>.json",
    )
    parser.add_argument(
        "--precomputed-dir",
        default=None,
        help="Default: config.PRECOMPUTED_DIR",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device",
        default=None,
        help="Examples: cuda, cuda:0, cpu. Default: config.DEVICE.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument("--pearson-tolerance", type=float, default=0.01)
    parser.add_argument("--pair-accuracy-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--pair-min-label-gap",
        type=float,
        default=None,
        help="Default: config.PAIRWISE_MIN_LABEL_GAP",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for final alpha selection.",
    )
    parser.add_argument(
        "--validate-precomputed-cache",
        action="store_true",
        help="Run the full current-config check on every precomputed sample.",
    )
    return parser.parse_args()


def read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    if size < 1:
        raise ValueError(f"Batch size must be positive, got {size}.")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def alpha_label(alpha: float) -> str:
    return f"{float(alpha):g}".replace("-", "m").replace(".", "p")


def alpha_score_column(alpha: float) -> str:
    return f"pred_ddg_alpha_{alpha_label(alpha)}"


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Pearson inputs must have the same length.")
    if len(xs) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )
    x_sum = sum((x - x_mean) ** 2 for x in xs)
    y_sum = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_sum * y_sum)
    return numerator / denominator if denominator > 0 else float("nan")


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        for sorted_position in range(position, end):
            ranks[indexed[sorted_position][0]] = average_rank
        position = end
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson_correlation(average_ranks(xs), average_ranks(ys))


def lower_is_better_auroc(
    labels: Sequence[float],
    predictions: Sequence[float],
    beneficial_threshold: float = 0.0,
) -> float:
    favorable = [
        prediction
        for label, prediction in zip(labels, predictions)
        if label < beneficial_threshold
    ]
    other = [
        prediction
        for label, prediction in zip(labels, predictions)
        if label >= beneficial_threshold
    ]
    if not favorable or not other:
        return float("nan")
    wins = 0.0
    comparisons = 0
    for favorable_score in favorable:
        for other_score in other:
            if favorable_score < other_score:
                wins += 1.0
            elif favorable_score == other_score:
                wins += 0.5
            comparisons += 1
    return wins / comparisons


def grouped_pair_accuracy(
    rows: Sequence[Dict[str, object]],
    predictions: Sequence[float],
    min_label_gap: float,
    same_site: Optional[bool],
) -> Tuple[float, int]:
    correct = 0.0
    pair_count = 0
    for left in range(len(rows)):
        left_complex = str(rows[left]["complex_id"])
        left_site = str(rows[left]["mutation_site_id"])
        left_label = float(rows[left]["label_ddg"])
        for right in range(left + 1, len(rows)):
            if str(rows[right]["complex_id"]) != left_complex:
                continue
            sites_match = str(rows[right]["mutation_site_id"]) == left_site
            if same_site is not None and sites_match != same_site:
                continue
            right_label = float(rows[right]["label_ddg"])
            label_difference = left_label - right_label
            if abs(label_difference) <= float(min_label_gap):
                continue
            prediction_difference = predictions[left] - predictions[right]
            product = label_difference * prediction_difference
            if product > 0:
                correct += 1.0
            elif prediction_difference == 0:
                correct += 0.5
            pair_count += 1
    if pair_count == 0:
        return float("nan"), 0
    return correct / pair_count, pair_count


def compute_alpha_metrics(
    rows: Sequence[Dict[str, object]],
    alpha: float,
    min_label_gap: float,
    beneficial_threshold: float = 0.0,
) -> Dict[str, object]:
    labels = [float(row["label_ddg"]) for row in rows]
    predictions = [
        float(row["pred_ddg_raw"])
        - float(alpha) * float(row["site_baseline_ddg"])
        for row in rows
    ]
    errors = [
        prediction - label
        for prediction, label in zip(predictions, labels)
    ]
    rmse = math.sqrt(mean([error * error for error in errors]))
    mae = mean([abs(error) for error in errors])
    cross_site_accuracy, cross_site_pairs = grouped_pair_accuracy(
        rows,
        predictions,
        min_label_gap=min_label_gap,
        same_site=False,
    )
    same_site_accuracy, same_site_pairs = grouped_pair_accuracy(
        rows,
        predictions,
        min_label_gap=min_label_gap,
        same_site=True,
    )
    baselines = [float(row["site_baseline_ddg"]) for row in rows]
    return {
        "alpha": float(alpha),
        "score_column": alpha_score_column(alpha),
        "samples": len(rows),
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson_correlation(labels, predictions),
        "spearman": spearman_correlation(labels, predictions),
        "beneficial_auroc": lower_is_better_auroc(
            labels,
            predictions,
            beneficial_threshold=beneficial_threshold,
        ),
        "cross_site_pair_accuracy": cross_site_accuracy,
        "cross_site_pair_count": cross_site_pairs,
        "same_site_pair_accuracy": same_site_accuracy,
        "same_site_pair_count": same_site_pairs,
        "prediction_mean": mean(predictions),
        "prediction_negative_ratio": (
            sum(prediction < beneficial_threshold for prediction in predictions)
            / len(predictions)
        ),
        "site_baseline_mean": mean(baselines),
        "site_baseline_mae": mean([abs(value) for value in baselines]),
    }


def _metric_satisfies(
    value: float,
    minimum: float,
) -> bool:
    if math.isnan(minimum):
        return True
    if math.isnan(value):
        return False
    return value >= minimum


def select_alpha(
    summaries: Sequence[Dict[str, object]],
    pearson_tolerance: float,
    pair_accuracy_tolerance: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    if not summaries:
        raise ValueError("No alpha summaries were provided.")
    baseline_rows = [
        row
        for row in summaries
        if math.isclose(float(row["alpha"]), 0.0, abs_tol=1e-12)
    ]
    if len(baseline_rows) != 1:
        raise ValueError(
            "Alpha sweep must contain exactly one alpha=0 baseline."
        )
    baseline = baseline_rows[0]
    minimum_pearson = float(baseline["pearson"]) - float(pearson_tolerance)
    minimum_pair_accuracy = (
        float(baseline["cross_site_pair_accuracy"])
        - float(pair_accuracy_tolerance)
    )
    eligible = [
        row
        for row in summaries
        if _metric_satisfies(float(row["pearson"]), minimum_pearson)
        and _metric_satisfies(
            float(row["cross_site_pair_accuracy"]),
            minimum_pair_accuracy,
        )
    ]
    if not eligible:
        selected = baseline
        reason = "No non-baseline alpha satisfied the constraints."
    else:
        selected = min(
            eligible,
            key=lambda row: (float(row["rmse"]), float(row["alpha"])),
        )
        reason = (
            "Minimum RMSE among alphas satisfying the Pearson and "
            "cross-site pair-accuracy constraints."
        )
    selection = {
        "selected_alpha": float(selected["alpha"]),
        "reason": reason,
        "baseline_alpha": float(baseline["alpha"]),
        "baseline_pearson": float(baseline["pearson"]),
        "minimum_allowed_pearson": minimum_pearson,
        "baseline_cross_site_pair_accuracy": float(
            baseline["cross_site_pair_accuracy"]
        ),
        "minimum_allowed_cross_site_pair_accuracy": minimum_pair_accuracy,
        "pearson_tolerance": float(pearson_tolerance),
        "pair_accuracy_tolerance": float(pair_accuracy_tolerance),
        "eligible_alphas": [float(row["alpha"]) for row in eligible],
        "selected_metrics": dict(selected),
    }
    return selected, selection


def make_noop_sample(sample: Dict[str, object]) -> Dict[str, object]:
    import torch

    noop = dict(sample)
    noop["mutant_antibody_seq"] = sample["antibody_seq"]
    noop["mutant_antigen_seq"] = sample["antigen_seq"]
    noop["delta_g"] = torch.zeros_like(sample["delta_g"])

    foldx_features = sample["foldx_features"].detach().cpu().reshape(-1)
    if int(foldx_features.numel()) != 3:
        raise ValueError(
            f"Expected three FoldX features, got {tuple(foldx_features.shape)}."
        )
    wt_energy = foldx_features[0]
    noop["foldx_energy"] = wt_energy.clone()
    noop["foldx_features"] = torch.stack(
        [wt_energy, wt_energy, torch.zeros_like(wt_energy)]
    )

    wt_esm_embedding = sample.get("wt_esm_embedding")
    if isinstance(wt_esm_embedding, torch.Tensor):
        noop["mut_esm_embedding"] = wt_esm_embedding.clone()
    mutation_esm_embedding = sample.get("mutation_esm_embedding")
    if isinstance(mutation_esm_embedding, torch.Tensor):
        noop["mutation_esm_embedding"] = torch.zeros_like(
            mutation_esm_embedding
        )
    wt_window = sample.get("wt_esm_window_tokens")
    if isinstance(wt_window, torch.Tensor):
        noop["mut_esm_window_tokens"] = wt_window.clone()

    structure_data = sample.get("structure_data")
    if not isinstance(structure_data, dict):
        raise ValueError("Self-baseline construction requires structure_data.")
    noop_structure = dict(structure_data)
    wt_aa_types = structure_data.get("wt_aa_types")
    if not isinstance(wt_aa_types, torch.Tensor):
        raise ValueError(
            "Self-baseline construction requires structure wt_aa_types."
        )
    noop_structure["mutant_aa_types"] = wt_aa_types.clone()
    mismatch_count = structure_data.get("mutation_wt_mismatch_count")
    if isinstance(mismatch_count, torch.Tensor):
        noop_structure["mutation_wt_mismatch_count"] = torch.zeros_like(
            mismatch_count
        )
    else:
        noop_structure["mutation_wt_mismatch_count"] = 0
    noop["structure_data"] = noop_structure
    return noop


def move_structure_to_device(structure_data, device):
    import torch

    if structure_data is None:
        return None
    for key, value in structure_data.items():
        if isinstance(value, torch.Tensor):
            structure_data[key] = value.to(device)
    return structure_data


def predict_samples(model, samples: List[Dict[str, object]], device) -> List[float]:
    import torch

    from data_loader import filter_none_collate

    batch = filter_none_collate(samples)
    foldx_energies = batch["foldx_energy"].to(device)
    foldx_features = batch["foldx_features"].to(device)
    structure_data = move_structure_to_device(
        batch.get("structure_data"),
        device,
    )
    tensor_keys = (
        "wt_esm_embedding",
        "mut_esm_embedding",
        "mutation_esm_embedding",
        "wt_esm_window_tokens",
        "mut_esm_window_tokens",
        "esm_window_padding_mask",
        "esm_window_mutation_mask",
    )
    model_inputs = {}
    for key in tensor_keys:
        value = batch.get(key)
        model_inputs[key] = (
            value.to(device)
            if isinstance(value, torch.Tensor)
            else value
        )
    with torch.inference_mode():
        predictions = model(
            antibody_seqs=batch["antibody_seq"],
            antigen_seqs=batch["antigen_seq"],
            mutant_antibody_seqs=batch["mutant_antibody_seq"],
            mutant_antigen_seqs=batch["mutant_antigen_seq"],
            foldx_energies=foldx_energies,
            structure_data=structure_data,
            foldx_features=foldx_features,
            **model_inputs,
        )
    return (
        predictions.detach()
        .float()
        .cpu()
        .reshape(-1)
        .tolist()
    )


def read_training_dataframe():
    import pandas as pd

    import config

    required_columns = [
        "#Pdb",
        "Mutation(s)_cleaned",
        "Affinity_mut_parsed",
        "Affinity_wt_parsed",
    ]
    errors = []
    for kwargs in ({"sep": ";"}, {"sep": None, "engine": "python"}):
        try:
            dataframe = pd.read_csv(config.CSV_PATH, **kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
            continue
        missing = set(required_columns).difference(dataframe.columns)
        if not missing:
            return dataframe.dropna(
                subset=required_columns
            ).reset_index(drop=True)
        errors.append(f"{kwargs}: missing columns {sorted(missing)}")
    raise RuntimeError(
        "Could not read the training dataframe: " + " | ".join(errors)
    )


def validation_positions(
    split_payload: Dict[str, object],
    dataset_size: int,
    max_samples: Optional[int],
) -> List[int]:
    expected_samples = int(split_payload.get("samples", dataset_size))
    if expected_samples != dataset_size:
        raise ValueError(
            "Split/dataset sample-count mismatch: "
            f"split={expected_samples}, dataset={dataset_size}."
        )
    positions = [int(value) for value in split_payload.get("val_indices", [])]
    if not positions:
        raise ValueError("Split JSON has no val_indices.")
    out_of_range = [
        position
        for position in positions
        if position < 0 or position >= dataset_size
    ]
    if out_of_range:
        raise IndexError(
            f"Validation positions are outside the dataset: {out_of_range[:10]}"
        )
    if max_samples is not None:
        positions = positions[: int(max_samples)]
    return positions


def infer_validation_predictions(
    ckpt_path: str,
    split_path: str,
    precomputed_dir: str,
    batch_size: int,
    device_name: Optional[str],
    max_samples: Optional[int],
    validate_precomputed_cache: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    import torch
    from tqdm import tqdm

    import config
    from case_study_unified import (
        build_model,
        load_weights,
        resolve_checkpoint_path,
    )
    from data_loader import PrecomputedDataset

    device = torch.device(device_name) if device_name else config.DEVICE
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config.DETERMINISTIC_STRUCTURE_SAMPLING = True
    dataset = PrecomputedDataset(
        precomputed_dir,
        validate_current_config=validate_precomputed_cache,
    )
    split_payload = read_json(split_path)
    positions = validation_positions(
        split_payload,
        dataset_size=len(dataset),
        max_samples=max_samples,
    )
    dataframe = read_training_dataframe()
    if dataset.indices and max(dataset.indices) >= len(dataframe):
        raise IndexError(
            f"Precomputed row {max(dataset.indices)} exceeds "
            f"training dataframe rows {len(dataframe)}."
        )

    resolved_checkpoint = resolve_checkpoint_path(ckpt_path)
    model = build_model(
        use_precomputed_esm=getattr(config, "USE_PRECOMPUTED_ESM", False)
    )
    model = load_weights(model, resolved_checkpoint, device)

    rows: List[Dict[str, object]] = []
    representative_position_by_site: Dict[str, int] = {}
    for position_batch in tqdm(
        list(chunks(positions, batch_size)),
        desc="Validation raw predictions",
        unit="batch",
    ):
        samples = [dataset[int(position)] for position in position_batch]
        predictions = predict_samples(model, samples, device)
        for position, sample, prediction in zip(
            position_batch,
            samples,
            predictions,
        ):
            source_index = int(dataset.indices[int(position)])
            source_row = dataframe.iloc[source_index]
            site_id = str(sample["mutation_site_id"])
            representative_position_by_site.setdefault(site_id, int(position))
            foldx_features = (
                sample["foldx_features"]
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )
            rows.append(
                {
                    "dataset_position": int(position),
                    "source_index": source_index,
                    "complex_id": str(sample["complex_id"]),
                    "mutation_site_id": site_id,
                    "pdb_id": str(sample.get("pdb_id", "")),
                    "mutation": str(source_row["Mutation(s)_cleaned"]),
                    "label_ddg": float(
                        sample["delta_g"].detach().cpu().reshape(-1)[0].item()
                    ),
                    "pred_ddg_raw": float(prediction),
                    "foldx_wt_energy": float(foldx_features[0]),
                    "foldx_mut_energy": float(foldx_features[1]),
                    "foldx_delta_interaction": float(foldx_features[2]),
                    "checkpoint": resolved_checkpoint,
                    "split_json": os.path.abspath(split_path),
                }
            )

    baseline_by_site: Dict[str, float] = {}
    representative_items = list(representative_position_by_site.items())
    for item_batch in tqdm(
        [
            representative_items[start : start + batch_size]
            for start in range(0, len(representative_items), batch_size)
        ],
        desc="Validation self baselines",
        unit="batch",
    ):
        site_ids = [site_id for site_id, _ in item_batch]
        noop_samples = [
            make_noop_sample(dataset[int(position)])
            for _, position in item_batch
        ]
        predictions = predict_samples(model, noop_samples, device)
        baseline_by_site.update(
            {
                site_id: float(prediction)
                for site_id, prediction in zip(site_ids, predictions)
            }
        )

    for row in rows:
        site_id = str(row["mutation_site_id"])
        if site_id not in baseline_by_site:
            raise KeyError(f"Missing self baseline for {site_id}.")
        row["site_baseline_ddg"] = baseline_by_site[site_id]

    metadata = {
        "checkpoint": resolved_checkpoint,
        "checkpoint_size": os.path.getsize(resolved_checkpoint),
        "checkpoint_mtime_ns": os.stat(resolved_checkpoint).st_mtime_ns,
        "split_json": os.path.abspath(split_path),
        "precomputed_dir": os.path.abspath(precomputed_dir),
        "validation_samples": len(rows),
        "unique_validation_sites": len(baseline_by_site),
        "limited_smoke_test": max_samples is not None,
        "batch_size": int(batch_size),
        "device": str(device),
        "deterministic_structure_sampling": True,
    }
    return rows, metadata


def print_summary(
    summaries: Sequence[Dict[str, object]],
    selected: Dict[str, object],
) -> None:
    print(
        "alpha\tRMSE\tMAE\tPearson\tSpearman\tAUROC\t"
        "CrossSiteAcc\tCrossPairs\tPred<0"
    )
    for row in summaries:
        print(
            f"{float(row['alpha']):.2f}\t"
            f"{float(row['rmse']):.6f}\t"
            f"{float(row['mae']):.6f}\t"
            f"{float(row['pearson']):.6f}\t"
            f"{float(row['spearman']):.6f}\t"
            f"{float(row['beneficial_auroc']):.6f}\t"
            f"{float(row['cross_site_pair_accuracy']):.6f}\t"
            f"{int(row['cross_site_pair_count'])}\t"
            f"{float(row['prediction_negative_ratio']):.4f}"
        )
    print(
        "\nSelected alpha: "
        f"{float(selected['alpha']):g} "
        f"(RMSE={float(selected['rmse']):.6f}, "
        f"Pearson={float(selected['pearson']):.6f}, "
        "cross-site accuracy="
        f"{float(selected['cross_site_pair_accuracy']):.6f})"
    )


def main() -> None:
    import config

    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1.")
    if args.pearson_tolerance < 0:
        raise ValueError("--pearson-tolerance must be non-negative.")
    if args.pair_accuracy_tolerance < 0:
        raise ValueError("--pair-accuracy-tolerance must be non-negative.")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    split_path = os.path.abspath(
        args.split_json
        or getattr(config, "SPLIT_JSON", "")
        or os.path.join(
            getattr(
                config,
                "SPLIT_DIR",
                os.path.join(config.VARIANT_DIR, "splits"),
            ),
            "complex_split_"
            f"{getattr(config, 'SPLIT_RATIO_TAG', '80_10_10')}_"
            f"seed{getattr(config, 'RANDOM_SEED', 42)}.json",
        )
    )
    precomputed_dir = os.path.abspath(
        args.precomputed_dir or config.PRECOMPUTED_DIR
    )
    checkpoint_path = os.path.abspath(args.ckpt or config.BEST_MODEL_PATH)
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Validation split JSON not found: {split_path}")
    if not os.path.isdir(precomputed_dir):
        raise FileNotFoundError(
            f"Precomputed sample directory not found: {precomputed_dir}"
        )

    rows, metadata = infer_validation_predictions(
        ckpt_path=checkpoint_path,
        split_path=split_path,
        precomputed_dir=precomputed_dir,
        batch_size=args.batch_size,
        device_name=args.device,
        max_samples=args.max_samples,
        validate_precomputed_cache=args.validate_precomputed_cache,
    )
    raw_predictions_path = os.path.join(
        out_dir,
        "validation_predictions_with_site_baseline.csv",
    )
    write_csv(raw_predictions_path, rows)
    write_json(
        os.path.join(out_dir, "validation_inference_metadata.json"),
        metadata,
    )

    alphas = list(dict.fromkeys(float(alpha) for alpha in args.alphas))
    if not any(math.isclose(alpha, 0.0, abs_tol=1e-12) for alpha in alphas):
        raise ValueError("--alphas must include 0 as the raw-score baseline.")
    min_label_gap = (
        float(args.pair_min_label_gap)
        if args.pair_min_label_gap is not None
        else float(getattr(config, "PAIRWISE_MIN_LABEL_GAP", 0.2))
    )
    beneficial_threshold = float(
        getattr(config, "BENEFICIAL_THRESHOLD", 0.0)
    )
    summaries = [
        compute_alpha_metrics(
            rows,
            alpha=alpha,
            min_label_gap=min_label_gap,
            beneficial_threshold=beneficial_threshold,
        )
        for alpha in alphas
    ]
    selected, selection = select_alpha(
        summaries,
        pearson_tolerance=args.pearson_tolerance,
        pair_accuracy_tolerance=args.pair_accuracy_tolerance,
    )
    selection.update(
        {
            "checkpoint": metadata["checkpoint"],
            "split_json": metadata["split_json"],
            "validation_samples": metadata["validation_samples"],
            "unique_validation_sites": metadata["unique_validation_sites"],
            "pair_min_label_gap": min_label_gap,
            "beneficial_threshold": beneficial_threshold,
            "warning": (
                "Do not use --max-samples results for final alpha selection."
                if args.max_samples is not None
                else ""
            ),
        }
    )

    scored_rows: List[Dict[str, object]] = []
    for row in rows:
        scored = dict(row)
        for alpha in alphas:
            scored[alpha_score_column(alpha)] = (
                float(row["pred_ddg_raw"])
                - alpha * float(row["site_baseline_ddg"])
            )
        scored_rows.append(scored)

    write_csv(
        os.path.join(out_dir, "validation_alpha_sweep_predictions.csv"),
        scored_rows,
    )
    write_csv(
        os.path.join(out_dir, "validation_alpha_sweep_summary.csv"),
        summaries,
    )
    write_json(
        os.path.join(out_dir, "validation_selected_alpha.json"),
        selection,
    )
    print_summary(summaries, selected)
    print(f"\nPredictions: {raw_predictions_path}")
    print(
        "Sweep      : "
        f"{os.path.join(out_dir, 'validation_alpha_sweep_summary.csv')}"
    )
    print(
        "Selected   : "
        f"{os.path.join(out_dir, 'validation_selected_alpha.json')}"
    )


if __name__ == "__main__":
    main()
