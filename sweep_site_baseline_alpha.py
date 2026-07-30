from __future__ import annotations

import argparse
import csv
import math
import os
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence


DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_TOPK = (20, 50, 100, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep partial self-mutation baseline correction on an existing "
            "antibody_opt_predictions.csv without rerunning inference."
        )
    )
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOPK),
    )
    return parser.parse_args()


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def alpha_label(alpha: float) -> str:
    text = f"{float(alpha):g}"
    return text.replace("-", "m").replace(".", "p")


def score_column(alpha: float) -> str:
    return f"pred_ddg_alpha_{alpha_label(alpha)}"


def _float(row: Dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(
            f"Missing required numeric column {key!r} for "
            f"mutation={row.get('mutation', '')!r}."
        )
    return float(value)


def validate_rows(rows: Sequence[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Prediction CSV is empty.")
    required = {"mutation", "is_favorable", "site_baseline_ddg"}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(
            "Prediction CSV was not generated with "
            f"--site-baseline-correction; missing columns: {missing}"
        )
    if "pred_ddg_raw" not in rows[0] and "pred_ddg" not in rows[0]:
        raise ValueError("Prediction CSV is missing pred_ddg_raw and pred_ddg.")


def raw_prediction(row: Dict[str, str]) -> float:
    key = "pred_ddg_raw" if str(row.get("pred_ddg_raw", "")).strip() else "pred_ddg"
    return _float(row, key)


def add_alpha_scores(
    rows: Sequence[Dict[str, str]],
    alphas: Iterable[float],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for row in rows:
        enriched: Dict[str, object] = dict(row)
        raw = raw_prediction(row)
        baseline = _float(row, "site_baseline_ddg")
        for alpha in alphas:
            enriched[score_column(alpha)] = raw - float(alpha) * baseline
        output.append(enriched)
    return output


def lower_is_better_auc(
    rows: Sequence[Dict[str, object]],
    score_col: str,
) -> float:
    favorable_scores = [
        float(row[score_col])
        for row in rows
        if int(row.get("is_favorable", 0)) == 1
    ]
    other_scores = [
        float(row[score_col])
        for row in rows
        if int(row.get("is_favorable", 0)) == 0
    ]
    if not favorable_scores or not other_scores:
        return float("nan")

    wins = 0.0
    comparisons = 0
    for favorable_score in favorable_scores:
        for other_score in other_scores:
            if favorable_score < other_score:
                wins += 1.0
            elif favorable_score == other_score:
                wins += 0.5
            comparisons += 1
    return wins / comparisons


def evaluate_alpha(
    rows: Sequence[Dict[str, object]],
    alpha: float,
    topk: Sequence[int],
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    score_col = score_column(alpha)
    ranked = sorted(rows, key=lambda row: float(row[score_col]))
    favorable_rows = [
        row for row in ranked if int(row.get("is_favorable", 0)) == 1
    ]
    favorable_count = len(favorable_rows)
    if favorable_count == 0:
        raise ValueError("No rows have is_favorable=1.")

    rank_by_mutation = {
        str(row["mutation"]): rank
        for rank, row in enumerate(ranked, start=1)
    }
    rank_percentiles = [
        rank_by_mutation[str(row["mutation"])] / len(ranked)
        for row in favorable_rows
    ]

    summary: Dict[str, object] = {
        "alpha": float(alpha),
        "score_column": score_col,
        "candidates": len(ranked),
        "favorable": favorable_count,
        "mean_rank_percentile": mean(rank_percentiles),
        "median_rank_percentile": median(rank_percentiles),
        "lower_is_better_auroc": lower_is_better_auc(ranked, score_col),
    }
    for k in topk:
        hit_count = sum(
            int(row.get("is_favorable", 0)) == 1
            for row in ranked[: min(int(k), len(ranked))]
        )
        summary[f"hit_count_at_{k}"] = hit_count
        summary[f"hits_at_{k}"] = hit_count / favorable_count

    favorable_details: List[Dict[str, object]] = []
    for row in favorable_rows:
        mutation = str(row["mutation"])
        rank = rank_by_mutation[mutation]
        favorable_details.append(
            {
                "alpha": float(alpha),
                "mutation": mutation,
                "rank": rank,
                "rank_percentile": rank / len(ranked),
                "pred_ddg_raw": raw_prediction(row),
                "site_baseline_ddg": _float(row, "site_baseline_ddg"),
                "score_alpha": float(row[score_col]),
                "score_column": score_col,
            }
        )
    favorable_details.sort(key=lambda row: int(row["rank"]))
    return summary, favorable_details


def print_results(
    summaries: Sequence[Dict[str, object]],
    favorable_details: Sequence[Dict[str, object]],
    topk: Sequence[int],
) -> None:
    hit_headers = [f"Hits@{k}" for k in topk]
    header = [
        "alpha",
        *hit_headers,
        "mean_pct",
        "median_pct",
        "AUROC",
    ]
    print("\t".join(header))
    for summary in summaries:
        values = [f"{float(summary['alpha']):.2f}"]
        values.extend(
            f"{int(summary[f'hit_count_at_{k}'])}/"
            f"{int(summary['favorable'])}"
            for k in topk
        )
        values.extend(
            [
                f"{float(summary['mean_rank_percentile']):.6f}",
                f"{float(summary['median_rank_percentile']):.6f}",
                (
                    "nan"
                    if math.isnan(float(summary["lower_is_better_auroc"]))
                    else f"{float(summary['lower_is_better_auroc']):.6f}"
                ),
            ]
        )
        print("\t".join(values))

    print("\nFavorable mutation ranks")
    current_alpha = None
    for row in favorable_details:
        alpha = float(row["alpha"])
        if alpha != current_alpha:
            current_alpha = alpha
            print(f"\nalpha={alpha:g}")
        print(
            f"{str(row['mutation']):>12s}  "
            f"rank={int(row['rank']):>4d}  "
            f"percentile={float(row['rank_percentile']):.4f}  "
            f"score={float(row['score_alpha']):.6f}"
        )


def main() -> None:
    args = parse_args()
    pred_csv = os.path.abspath(args.pred_csv)
    out_dir = os.path.abspath(
        args.out_dir or os.path.dirname(pred_csv) or "."
    )
    alphas = list(dict.fromkeys(float(alpha) for alpha in args.alphas))
    topk = list(dict.fromkeys(int(k) for k in args.topk))
    if any(k < 1 for k in topk):
        raise ValueError(f"All top-k values must be positive: {topk}")

    rows = read_rows(pred_csv)
    validate_rows(rows)
    scored_rows = add_alpha_scores(rows, alphas)

    summaries: List[Dict[str, object]] = []
    favorable_details: List[Dict[str, object]] = []
    for alpha in alphas:
        summary, details = evaluate_alpha(scored_rows, alpha, topk)
        summaries.append(summary)
        favorable_details.extend(details)

    predictions_out = os.path.join(
        out_dir,
        "antibody_opt_predictions_alpha_sweep.csv",
    )
    summary_out = os.path.join(
        out_dir,
        "antibody_opt_alpha_sweep_summary.csv",
    )
    ranks_out = os.path.join(
        out_dir,
        "antibody_opt_alpha_sweep_favorable_ranks.csv",
    )
    write_rows(predictions_out, scored_rows)
    write_rows(summary_out, summaries)
    write_rows(ranks_out, favorable_details)
    print_results(summaries, favorable_details, topk)
    print(f"\nPredictions: {predictions_out}")
    print(f"Summary    : {summary_out}")
    print(f"Ranks      : {ranks_out}")


if __name__ == "__main__":
    main()
