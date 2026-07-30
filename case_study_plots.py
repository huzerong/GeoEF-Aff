import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots for case-study predictions.")
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--task", choices=["regression", "ranking"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--score-col",
        default="pred_ddg",
        help="Prediction column used for ranking plots.",
    )
    return parser.parse_args()


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_regression(rows, out_dir):
    valid_rows = [r for r in rows if r.get("experimental_ddg", "") != ""]
    y_true = np.array([float(r["experimental_ddg"]) for r in valid_rows], dtype=float)
    y_pred = np.array([float(r["pred_ddg"]) for r in valid_rows], dtype=float)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.65)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Experimental ddG")
    plt.ylabel("Predicted ddG")
    plt.title("Experimental vs Predicted ddG")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_exp_vs_pred.png"), dpi=300)
    plt.close()


def plot_ranking(rows, out_dir, score_col="pred_ddg"):
    ranked = sorted(rows, key=lambda x: float(x[score_col]))
    favorable = [r for r in ranked if int(r.get("is_favorable", "0")) == 1]

    muts = [r["mutation"] for r in favorable]
    ranks = [ranked.index(r) + 1 for r in favorable]

    plt.figure(figsize=(8, 4))
    plt.bar(muts, ranks)
    plt.ylabel("Rank (lower is better)")
    plt.title("Ranks of favorable mutations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "favorable_mutation_ranks.png"), dpi=300)
    plt.close()

    xs = list(range(1, len(ranked) + 1))
    ys = []
    total_fav = len(favorable)
    hit_count = 0
    for i, row in enumerate(ranked, start=1):
        if int(row.get("is_favorable", "0")) == 1:
            hit_count += 1
        ys.append(hit_count / total_fav if total_fav else 0.0)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.xlabel("Top-k")
    plt.ylabel("Hits@k")
    plt.title("Enrichment curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hits_at_k_curve.png"), dpi=300)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_rows(args.pred_csv)

    if args.task == "regression":
        plot_regression(rows, args.out_dir)
    else:
        plot_ranking(rows, args.out_dir, score_col=args.score_col)


if __name__ == "__main__":
    main()
