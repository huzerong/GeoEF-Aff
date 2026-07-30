import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd

import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an explicit training CSV containing only rows whose "
            "mutation-specific FoldX cache is status=ok."
        )
    )
    parser.add_argument("--csv", default=config.CSV_PATH, help="Input SKEMPI-style CSV.")
    parser.add_argument(
        "--cache-dir",
        default=config.MUTATION_FOLDX_CACHE_DIR,
        help="Mutation FoldX cache directory containing sample_json/<row_idx>.json.",
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join(config.VARIANT_DIR, "skempi_v2_foldx_ok.csv"),
        help="Filtered training CSV path.",
    )
    parser.add_argument(
        "--rejected-csv",
        default=os.path.join(config.VARIANT_DIR, "skempi_v2_foldx_rejected.csv"),
        help="Rejected rows report path.",
    )
    parser.add_argument(
        "--summary-json",
        default=os.path.join(config.VARIANT_DIR, "skempi_v2_foldx_filter_summary.json"),
        help="Filtering summary JSON path.",
    )
    return parser.parse_args()


def load_cache(cache_dir: str, row_idx: int) -> Tuple[str, str]:
    path = os.path.join(cache_dir, "sample_json", f"{row_idx}.json")
    if not os.path.exists(path):
        return "missing", f"cache file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return "invalid_json", repr(exc)

    status = str(payload.get("status", "missing_status"))
    if status != "ok":
        return status, str(payload.get("error", ""))

    required = ("wt_energy", "mut_energy", "foldx_delta_interaction")
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        return "missing_fields", ",".join(missing)
    return "ok", ""


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv, sep=";")
    df = df.dropna(subset=["#Pdb", "Mutation(s)_cleaned"]).copy()

    keep_indices: List[int] = []
    rejected_rows: List[Dict[str, object]] = []
    status_counts: Dict[str, int] = {}

    for row_idx, row in df.iterrows():
        status, reason = load_cache(args.cache_dir, int(row_idx))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "ok":
            keep_indices.append(row_idx)
            continue
        rejected_rows.append(
            {
                "row_idx": int(row_idx),
                "status": status,
                "reason": reason,
                "#Pdb": row.get("#Pdb", ""),
                "Mutation(s)_cleaned": row.get("Mutation(s)_cleaned", ""),
            }
        )

    filtered = df.loc[keep_indices].copy()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.rejected_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json) or ".", exist_ok=True)

    # Keep the original DataFrame index in the file so future audits can map rows
    # back to sample_json/<row_idx>.json even after filtering.
    filtered.to_csv(args.out_csv, sep=";", index=True, index_label="original_row_idx")
    pd.DataFrame(rejected_rows).to_csv(args.rejected_csv, index=False)

    summary = {
        "input_csv": os.path.abspath(args.csv),
        "cache_dir": os.path.abspath(args.cache_dir),
        "out_csv": os.path.abspath(args.out_csv),
        "rejected_csv": os.path.abspath(args.rejected_csv),
        "input_rows": int(len(df)),
        "kept_rows": int(len(filtered)),
        "rejected_rows": int(len(rejected_rows)),
        "status_counts": status_counts,
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

