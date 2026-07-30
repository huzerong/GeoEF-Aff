from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
RUN_ALL_SCRIPT = os.path.join(SCRIPT_DIR, "run_all_case_studies.py")

import config

CASE_STUDY_TASKS = ("rbd_ddg", "antibody_opt")
UNIAIR_TASKS = ("HER2", "TCR_pMHC_Atlas")
ALL_TASKS = CASE_STUDY_TASKS + UNIAIR_TASKS

DEFAULT_UNIAIR_FOLDX_REUSE_ROOTS = (
    os.path.join(config.DATA_DIR, "external_uniair", "foldx_3feature_cache"),
    os.path.join(
        config.DATA_DIR,
        "external_uniair",
        "foldx_3feature_esm650m_hd384_raad3_rankloss01_cache",
    ),
    os.path.join(config.OUTPUT_DIR, "case_studies", "all_tasks_fixed"),
)

CHECKPOINT_RE = re.compile(
    r"^epoch_(?P<epoch>\d+)_pearson_"
    r"(?P<pearson>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\.pth$"
)

BASE_COLUMNS = [
    "checkpoint",
    "checkpoint_path",
    "epoch",
    "validation_pearson",
    "status",
    "completed_tasks",
    "missing_tasks",
    "runner_returncode",
    "output_dir",
]

PREFERRED_METRIC_ORDER = {
    "rbd_ddg": [
        "samples",
        "drop_zero_labels",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "auroc",
    ],
    "antibody_opt": [
        "candidates",
        "favorable",
        "hits_at_10",
        "hits_at_20",
        "hits_at_50",
        "hits_at_100",
        "mean_rank_percentile",
        "median_rank_percentile",
        "mean_within_site_rank",
        "median_within_site_rank",
        "mean_within_site_rank_percentile",
        "within_site_hits_at_1",
        "within_site_hits_at_3",
        "within_site_hits_at_5",
    ],
    "HER2": [
        "samples",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "rmse_calibrated",
        "mae_calibrated",
        "auroc",
        "calibration_slope",
        "calibration_intercept",
        "loss_mse",
    ],
    "TCR_pMHC_Atlas": [
        "samples",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "rmse_calibrated",
        "mae_calibrated",
        "auroc",
        "calibration_slope",
        "calibration_intercept",
        "loss_mse",
    ],
}


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    name: str
    stem: str
    epoch: int
    validation_pearson: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every epoch checkpoint whose filename Pearson reaches a threshold, "
            "then collect all task metrics into one wide CSV."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join(
            SCRIPT_DIR,
            "training_checkpoints_hd384_raad3_beneficial_grouprank_muttype_localtoken32",
        ),
    )
    parser.add_argument("--checkpoint-glob", default="epoch_*_pearson_*.pth")
    parser.add_argument("--min-pearson", type=float, default=0.69)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Sweep output root. Each checkpoint gets one subdirectory.",
    )
    parser.add_argument("--summary-csv", default=None, help="Final wide CSV path.")
    parser.add_argument(
        "--tasks",
        default=",".join(ALL_TASKS),
        help=f"Comma-separated subset of: {', '.join(ALL_TASKS)}.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Visible GPU IDs. Four tasks use four GPUs per checkpoint.",
    )
    parser.add_argument(
        "--checkpoints-in-parallel",
        type=int,
        default=None,
        help="Default: floor(number of GPUs / number of tasks). With 4 GPUs and 4 tasks this is 1.",
    )
    parser.add_argument(
        "--case-prepared-cache-dir",
        default=config.CASE_STUDY_PREPARED_CACHE_DIR,
    )
    parser.add_argument(
        "--case-skip-wt-aa-check",
        action="store_true",
        help="Relax WT amino-acid checks for rbd_ddg/antibody_opt prepared inputs.",
    )
    parser.add_argument(
        "--case-site-baseline-correction",
        action="store_true",
        help="Use self-mutation site baselines for antibody_opt ranking.",
    )
    parser.add_argument(
        "--uniair-precomputed-root",
        default=None,
        help="Shared root laid out as <root>/<dataset>/precomputed.",
    )
    parser.add_argument("--her2-precomputed-dir", default=None)
    parser.add_argument("--tcr-pmhc-precomputed-dir", default=None)
    parser.add_argument("--uniair-data-root", default=None)
    parser.add_argument("--uniair-reuse-foldx-json-root", default=None)
    parser.add_argument("--her2-reuse-foldx-json-dir", default=None)
    parser.add_argument("--tcr-pmhc-reuse-foldx-json-dir", default=None)
    parser.add_argument("--her2-csv", default=None)
    parser.add_argument("--tcr-pmhc-csv", default=None)
    parser.add_argument("--uniair-batch-size", type=int, default=2)
    parser.add_argument("--uniair-num-workers", type=int, default=0)
    parser.add_argument("--uniair-esm-batch-size", type=int, default=8)
    parser.add_argument("--uniair-foldx-workers", type=int, default=4)
    parser.add_argument("--uniair-allow-zero-energy", action="store_true")
    parser.add_argument("--uniair-skip-wt-aa-check", action="store_true")
    parser.add_argument("--uniair-strict-tcr-wt-aa-check", action="store_true")
    parser.add_argument("--uniair-skip-mut-pdb-sequence-check", action="store_true")
    parser.add_argument("--with-plots", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun checkpoints even when all summaries exist.")
    parser.add_argument(
        "--force-full-uniair-pipeline",
        action="store_true",
        help="Run UniAIR preparation for every checkpoint instead of direct evaluation after caches exist.",
    )
    parser.add_argument(
        "--no-cache-warmup",
        action="store_true",
        help="Do not run the first pending checkpoint alone before parallel checkpoint evaluation.",
    )
    parser.add_argument("--collect-only", action="store_true", help="Only rebuild the summary CSV from existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def parse_tasks(raw: str) -> List[str]:
    tasks: List[str] = []
    for value in raw.split(","):
        task = value.strip()
        if task and task not in tasks:
            tasks.append(task)
    unknown = [task for task in tasks if task not in ALL_TASKS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Valid tasks: {list(ALL_TASKS)}")
    if not tasks:
        raise ValueError("At least one task is required.")
    return tasks


def parse_gpus(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def discover_checkpoints(
    checkpoint_dir: str,
    checkpoint_glob: str,
    min_pearson: float,
) -> List[CheckpointInfo]:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    found: List[CheckpointInfo] = []
    ignored: List[str] = []
    for path in glob.glob(os.path.join(checkpoint_dir, checkpoint_glob)):
        name = os.path.basename(path)
        match = CHECKPOINT_RE.match(name)
        if not match:
            ignored.append(name)
            continue
        pearson = float(match.group("pearson"))
        if pearson + 1e-12 < min_pearson:
            continue
        found.append(
            CheckpointInfo(
                path=os.path.abspath(path),
                name=name,
                stem=os.path.splitext(name)[0],
                epoch=int(match.group("epoch")),
                validation_pearson=pearson,
            )
        )
    found.sort(key=lambda item: (item.epoch, item.name))
    if ignored:
        print(f"Ignored {len(ignored)} checkpoint files with unrecognized names.")
    if not found:
        raise FileNotFoundError(
            f"No checkpoints with filename Pearson >= {min_pearson:.4f} found in "
            f"{checkpoint_dir!r} using {checkpoint_glob!r}."
        )
    return found


def checkpoint_output_dir(out_root: str, checkpoint: CheckpointInfo) -> str:
    return os.path.join(out_root, checkpoint.stem)


def task_summary_path(checkpoint_dir: str, task: str) -> str:
    if task in CASE_STUDY_TASKS:
        return os.path.join(checkpoint_dir, task, f"{task}_summary.csv")
    return os.path.join(checkpoint_dir, task, "eval", "metrics.csv")


def task_is_complete(checkpoint_dir: str, task: str) -> bool:
    path = task_summary_path(checkpoint_dir, task)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def checkpoint_is_complete(checkpoint_dir: str, tasks: Sequence[str]) -> bool:
    return all(task_is_complete(checkpoint_dir, task) for task in tasks)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_task_metrics(checkpoint_dir: str, task: str) -> Optional[Dict[str, str]]:
    path = task_summary_path(checkpoint_dir, task)
    if not os.path.isfile(path):
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None
    if task in UNIAIR_TASKS:
        for row in rows:
            if row.get("method") == "Ours_3Feature":
                return row
        for row in rows:
            if row.get("checkpoint") != "FoldX":
                return row
    return rows[0]


def metric_prefix(task: str) -> str:
    return "TCR_pMHC" if task == "TCR_pMHC_Atlas" else task


def collect_rows(
    checkpoints: Sequence[CheckpointInfo],
    tasks: Sequence[str],
    out_root: str,
    returncodes: Optional[Dict[str, int]] = None,
) -> List[Dict[str, object]]:
    returncodes = returncodes or {}
    collected: List[Dict[str, object]] = []
    for checkpoint in checkpoints:
        ckpt_out = checkpoint_output_dir(out_root, checkpoint)
        row: Dict[str, object] = {
            "checkpoint": checkpoint.name,
            "checkpoint_path": checkpoint.path,
            "epoch": checkpoint.epoch,
            "validation_pearson": checkpoint.validation_pearson,
            "runner_returncode": returncodes.get(checkpoint.path, ""),
            "output_dir": ckpt_out,
        }
        completed: List[str] = []
        missing: List[str] = []
        for task in tasks:
            metrics = read_task_metrics(ckpt_out, task)
            if metrics is None:
                missing.append(task)
                continue
            completed.append(task)
            prefix = metric_prefix(task)
            preferred = PREFERRED_METRIC_ORDER.get(task, [])
            ordered_keys = preferred + [key for key in metrics if key not in preferred]
            for key in ordered_keys:
                if key in {"dataset", "method", "checkpoint"} or key not in metrics:
                    continue
                row[f"{prefix}_{key}"] = metrics[key]
        row["completed_tasks"] = ",".join(completed)
        row["missing_tasks"] = ",".join(missing)
        row["status"] = "ok" if not missing else ("partial" if completed else "missing")
        collected.append(row)
    return collected


def write_summary_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(BASE_COLUMNS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def resolve_uniair_precomputed_dir(
    dataset: str,
    args: argparse.Namespace,
    out_root: str,
) -> str:
    explicit = (
        args.her2_precomputed_dir
        if dataset == "HER2"
        else args.tcr_pmhc_precomputed_dir
    )
    if explicit:
        return os.path.abspath(explicit)
    if args.uniair_precomputed_root:
        return os.path.abspath(
            os.path.join(args.uniair_precomputed_root, dataset, "precomputed")
        )

    candidates = [
        os.path.join(
            config.OUTPUT_DIR,
            "case_studies",
            "all_tasks_beneficial_grouprank_muttype_localtoken32",
            dataset,
            "precomputed",
        ),
        os.path.join(
            config.DATA_DIR,
            "external_uniair",
            "precomputed",
            f"{dataset}_3feature_esm650m_hd384_raad3_beneficial_grouprank_muttype_localtoken32",
        ),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(out_root, "_shared_cache", dataset, "precomputed"))


def has_pt_samples(path: str) -> bool:
    return os.path.isdir(path) and any(name.endswith(".pt") for name in os.listdir(path))


def has_foldx_json_samples(path: str) -> bool:
    return os.path.isdir(path) and any(
        name.endswith(".json") for name in os.listdir(path)
    )


def auto_configure_uniair_foldx_reuse(
    args: argparse.Namespace,
    tasks: Sequence[str],
) -> None:
    requested = [task for task in tasks if task in UNIAIR_TASKS]
    if not requested:
        return
    if (
        args.uniair_reuse_foldx_json_root
        or args.her2_reuse_foldx_json_dir
        or args.tcr_pmhc_reuse_foldx_json_dir
    ):
        return

    for candidate in DEFAULT_UNIAIR_FOLDX_REUSE_ROOTS:
        if all(
            has_foldx_json_samples(os.path.join(candidate, dataset, "sample_json"))
            for dataset in requested
        ):
            args.uniair_reuse_foldx_json_root = os.path.abspath(candidate)
            return


def resolved_uniair_foldx_reuse_dirs(
    args: argparse.Namespace,
    tasks: Sequence[str],
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for dataset in tasks:
        if dataset not in UNIAIR_TASKS:
            continue
        specific = (
            args.her2_reuse_foldx_json_dir
            if dataset == "HER2"
            else args.tcr_pmhc_reuse_foldx_json_dir
        )
        if specific:
            resolved[dataset] = os.path.abspath(specific)
        elif args.uniair_reuse_foldx_json_root:
            resolved[dataset] = os.path.abspath(
                os.path.join(
                    args.uniair_reuse_foldx_json_root,
                    dataset,
                    "sample_json",
                )
            )
    return resolved


def maybe_add_value(command: List[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def maybe_add_flag(command: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def build_run_command(
    checkpoint: CheckpointInfo,
    checkpoint_out: str,
    tasks: Sequence[str],
    gpu_group: Sequence[str],
    args: argparse.Namespace,
    precomputed_dirs: Dict[str, str],
    uniair_eval_only: bool,
) -> List[str]:
    command = [
        args.python,
        RUN_ALL_SCRIPT,
        "--ckpt",
        checkpoint.path,
        "--out-dir",
        checkpoint_out,
        "--tasks",
        ",".join(tasks),
        "--unsafe-skip-confirmation",
        "--python",
        args.python,
    ]
    if gpu_group:
        command.extend(["--gpus", ",".join(gpu_group)])
    if not args.with_plots:
        command.append("--skip-plots")

    maybe_add_value(command, "--case-prepared-cache-dir", args.case_prepared_cache_dir)
    maybe_add_flag(command, "--case-skip-wt-aa-check", args.case_skip_wt_aa_check)
    maybe_add_flag(
        command,
        "--case-site-baseline-correction",
        args.case_site_baseline_correction,
    )
    maybe_add_value(command, "--uniair-data-root", args.uniair_data_root)
    maybe_add_value(command, "--uniair-reuse-foldx-json-root", args.uniair_reuse_foldx_json_root)
    maybe_add_value(command, "--her2-reuse-foldx-json-dir", args.her2_reuse_foldx_json_dir)
    maybe_add_value(command, "--tcr-pmhc-reuse-foldx-json-dir", args.tcr_pmhc_reuse_foldx_json_dir)
    maybe_add_value(command, "--her2-csv", args.her2_csv)
    maybe_add_value(command, "--tcr-pmhc-csv", args.tcr_pmhc_csv)
    maybe_add_value(command, "--uniair-batch-size", args.uniair_batch_size)
    maybe_add_value(command, "--uniair-num-workers", args.uniair_num_workers)
    maybe_add_value(command, "--uniair-esm-batch-size", args.uniair_esm_batch_size)
    maybe_add_value(command, "--uniair-foldx-workers", args.uniair_foldx_workers)
    maybe_add_flag(command, "--uniair-allow-zero-energy", args.uniair_allow_zero_energy)
    maybe_add_flag(command, "--uniair-skip-wt-aa-check", args.uniair_skip_wt_aa_check)
    maybe_add_flag(
        command,
        "--uniair-strict-tcr-wt-aa-check",
        args.uniair_strict_tcr_wt_aa_check,
    )
    maybe_add_flag(
        command,
        "--uniair-skip-mut-pdb-sequence-check",
        args.uniair_skip_mut_pdb_sequence_check,
    )
    if "HER2" in tasks:
        command.extend(["--her2-precomputed-dir", precomputed_dirs["HER2"]])
    if "TCR_pMHC_Atlas" in tasks:
        command.extend(
            ["--tcr-pmhc-precomputed-dir", precomputed_dirs["TCR_pMHC_Atlas"]]
        )
    maybe_add_flag(command, "--uniair-eval-only", uniair_eval_only)
    return command


def run_checkpoint(
    checkpoint: CheckpointInfo,
    tasks: Sequence[str],
    gpu_group: Sequence[str],
    args: argparse.Namespace,
    out_root: str,
    precomputed_dirs: Dict[str, str],
    uniair_eval_only: bool,
) -> int:
    checkpoint_out = checkpoint_output_dir(out_root, checkpoint)
    os.makedirs(checkpoint_out, exist_ok=True)
    command = build_run_command(
        checkpoint,
        checkpoint_out,
        tasks,
        gpu_group,
        args,
        precomputed_dirs,
        uniair_eval_only,
    )
    print(
        f"[START] epoch={checkpoint.epoch:03d} val_pearson={checkpoint.validation_pearson:.4f} "
        f"gpus={','.join(gpu_group) if gpu_group else 'sequential'}"
    )
    if args.dry_run:
        print(shlex.join(command))
        return 0

    log_path = os.path.join(checkpoint_out, "checkpoint_sweep.log")
    started = time.time()
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write("\n" + "=" * 100 + "\n")
        log_handle.write(f"command: {shlex.join(command)}\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    label = "OK" if completed.returncode == 0 else f"FAIL rc={completed.returncode}"
    print(
        f"[{label}] epoch={checkpoint.epoch:03d} elapsed={elapsed / 60:.1f}m "
        f"log={log_path}"
    )
    return int(completed.returncode)


def partition_gpu_groups(
    gpus: Sequence[str],
    task_count: int,
    requested_parallelism: Optional[int],
) -> List[List[str]]:
    if not gpus:
        if requested_parallelism not in (None, 1):
            raise ValueError("--checkpoints-in-parallel > 1 requires --gpus.")
        return [[]]
    maximum = len(gpus) // task_count
    if maximum < 1:
        raise ValueError(
            f"Need at least {task_count} GPUs for {task_count} parallel tasks; got {list(gpus)}."
        )
    parallelism = requested_parallelism or maximum
    if parallelism < 1 or parallelism > maximum:
        raise ValueError(
            f"--checkpoints-in-parallel must be between 1 and {maximum} for "
            f"{len(gpus)} GPUs and {task_count} tasks."
        )
    return [
        list(gpus[index * task_count : (index + 1) * task_count])
        for index in range(parallelism)
    ]


def print_selection(
    checkpoints: Sequence[CheckpointInfo],
    tasks: Sequence[str],
    gpu_groups: Sequence[Sequence[str]],
    out_root: str,
    summary_csv: str,
    precomputed_dirs: Dict[str, str],
    foldx_reuse_dirs: Dict[str, str],
) -> None:
    print("=" * 88)
    print("CHECKPOINT SWEEP")
    print(f"Selected checkpoints : {len(checkpoints)}")
    print(
        "Epochs               : "
        + ", ".join(
            f"{checkpoint.epoch}({checkpoint.validation_pearson:.4f})"
            for checkpoint in checkpoints
        )
    )
    print(f"Tasks                : {', '.join(tasks)}")
    print(
        "GPU groups           : "
        + " | ".join(",".join(group) if group else "sequential" for group in gpu_groups)
    )
    for dataset, path in precomputed_dirs.items():
        print(f"{dataset} precomputed  : {path}")
    if foldx_reuse_dirs:
        for dataset, path in foldx_reuse_dirs.items():
            print(f"{dataset} FoldX reuse : {path}")
    elif any(task in UNIAIR_TASKS for task in tasks):
        print("FoldX JSON reuse     : disabled (missing or not requested)")
    print(f"Output root          : {out_root}")
    print(f"Summary CSV          : {summary_csv}")
    print("=" * 88)


def refresh_summary(
    summary_csv: str,
    checkpoints: Sequence[CheckpointInfo],
    tasks: Sequence[str],
    out_root: str,
    returncodes: Dict[str, int],
) -> None:
    rows = collect_rows(checkpoints, tasks, out_root, returncodes=returncodes)
    write_summary_csv(summary_csv, rows)
    completed = sum(row["status"] == "ok" for row in rows)
    print(f"Summary updated: {summary_csv} ({completed}/{len(rows)} checkpoints complete)")


def run_batches(
    checkpoints: Sequence[CheckpointInfo],
    selected_checkpoints: Sequence[CheckpointInfo],
    gpu_groups: Sequence[Sequence[str]],
    tasks: Sequence[str],
    args: argparse.Namespace,
    out_root: str,
    precomputed_dirs: Dict[str, str],
    summary_csv: str,
    returncodes: Dict[str, int],
    uniair_eval_only: bool,
) -> None:
    width = len(gpu_groups)
    for offset in range(0, len(checkpoints), width):
        batch = list(checkpoints[offset : offset + width])
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    run_checkpoint,
                    checkpoint,
                    tasks,
                    gpu_groups[index],
                    args,
                    out_root,
                    precomputed_dirs,
                    uniair_eval_only,
                ): checkpoint
                for index, checkpoint in enumerate(batch)
            }
            for future in as_completed(futures):
                checkpoint = futures[future]
                try:
                    returncodes[checkpoint.path] = int(future.result())
                except Exception as exc:
                    returncodes[checkpoint.path] = -1
                    print(f"[FAIL] {checkpoint.name}: {exc}")
        refresh_summary(
            summary_csv,
            checkpoints=selected_checkpoints,
            tasks=tasks,
            out_root=out_root,
            returncodes=returncodes,
        )


def main() -> None:
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    auto_configure_uniair_foldx_reuse(args, tasks)
    gpus = parse_gpus(args.gpus)
    checkpoints = discover_checkpoints(
        args.checkpoint_dir,
        args.checkpoint_glob,
        args.min_pearson,
    )
    threshold_tag = f"{args.min_pearson:.4f}".replace(".", "p")
    out_root = os.path.abspath(
        args.out_dir
        or os.path.join(
            config.OUTPUT_DIR,
            "checkpoint_sweeps",
            f"checkpoint_sweep_ge_{threshold_tag}",
        )
    )
    summary_csv = os.path.abspath(
        args.summary_csv or os.path.join(out_root, "checkpoint_sweep_summary.csv")
    )
    os.makedirs(out_root, exist_ok=True)

    gpu_groups = partition_gpu_groups(gpus, len(tasks), args.checkpoints_in_parallel)
    precomputed_dirs = {
        dataset: resolve_uniair_precomputed_dir(dataset, args, out_root)
        for dataset in tasks
        if dataset in UNIAIR_TASKS
    }
    foldx_reuse_dirs = resolved_uniair_foldx_reuse_dirs(args, tasks)
    print_selection(
        checkpoints,
        tasks,
        gpu_groups,
        out_root,
        summary_csv,
        precomputed_dirs,
        foldx_reuse_dirs,
    )

    returncodes: Dict[str, int] = {}
    refresh_summary(summary_csv, checkpoints, tasks, out_root, returncodes)
    if args.collect_only:
        return

    pending = [
        checkpoint
        for checkpoint in checkpoints
        if args.force
        or not checkpoint_is_complete(
            checkpoint_output_dir(out_root, checkpoint),
            tasks,
        )
    ]
    if not pending:
        print("All selected checkpoints already have complete summaries.")
        return

    has_uniair = any(task in UNIAIR_TASKS for task in tasks)
    uniair_cache_ready = all(
        has_pt_samples(path) for path in precomputed_dirs.values()
    ) if has_uniair else False
    eval_only = (
        args.no_cache_warmup
        and has_uniair
        and uniair_cache_ready
        and not args.force_full_uniair_pipeline
    )

    should_warm = (
        bool(pending)
        and not args.no_cache_warmup
        and (
            (len(gpu_groups) > 1 and len(pending) > 1)
            or (has_uniair and not args.force_full_uniair_pipeline)
        )
    )
    if should_warm:
        warmup = pending.pop(0)
        returncodes[warmup.path] = run_checkpoint(
            warmup,
            tasks,
            gpu_groups[0],
            args,
            out_root,
            precomputed_dirs,
            False,
        )
        refresh_summary(summary_csv, checkpoints, tasks, out_root, returncodes)
        if has_uniair and not args.force_full_uniair_pipeline:
            eval_only = all(has_pt_samples(path) for path in precomputed_dirs.values())

    if pending:
        run_batches(
            pending,
            checkpoints,
            gpu_groups,
            tasks,
            args,
            out_root,
            precomputed_dirs,
            summary_csv,
            returncodes,
            eval_only,
        )

    refresh_summary(summary_csv, checkpoints, tasks, out_root, returncodes)
    if args.dry_run:
        print("Dry run complete; no checkpoint inference was executed.")
        return
    failed = [
        checkpoint.name
        for checkpoint in checkpoints
        if not checkpoint_is_complete(
            checkpoint_output_dir(out_root, checkpoint),
            tasks,
        )
    ]
    if failed:
        print(f"Incomplete checkpoints: {', '.join(failed)}")
        sys.exit(1)
    print("Checkpoint sweep complete.")


if __name__ == "__main__":
    main()
