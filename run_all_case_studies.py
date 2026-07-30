from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

import config

CASE_STUDY_UNIFIED = os.path.join(SCRIPT_DIR, "case_study_unified.py")
UNIAIR_RUN_PIPELINE = os.path.join(SCRIPT_DIR, "uniair_external", "run_pipeline.py")
UNIAIR_EVAL = os.path.join(SCRIPT_DIR, "uniair_external", "eval_uniair_external_3feature.py")

CASE_STUDY_TASKS = ("rbd_ddg", "antibody_opt")
UNIAIR_TASKS = ("HER2", "TCR_pMHC_Atlas")
ALL_TASKS = CASE_STUDY_TASKS + UNIAIR_TASKS

TASK_DISPLAY = {
    "rbd_ddg": "RBD ddG           (6M0J regression)",
    "antibody_opt": "Antibody Opt       (7FAE_RBD_Fv ranking)",
    "HER2": "UniAIR HER2         (external eval)",
    "TCR_pMHC_Atlas": "UniAIR TCR-pMHC     (external eval)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Total entry point: chain all four case-study tasks (RBD ddG, antibody_opt, HER2, TCR-pMHC)."
    )
    parser.add_argument(
        "--ckpt",
        default=config.BEST_MODEL_PATH,
        help="Checkpoint path shared across all tasks. Default filename: best_model.pth.",
    )
    parser.add_argument("--out-dir", default=None, help="Base output directory. Individual tasks write to subdirectories.")
    parser.add_argument(
        "--tasks",
        default=None,
        help=f"Comma-separated list of tasks. Choices: {', '.join(ALL_TASKS)}. Default: all four.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU IDs, e.g. 0,1,2,3. When set, tasks run in parallel, one per GPU. "
             "When unset, tasks run sequentially on the default device.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation for case_study tasks.")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metric computation for case_study tasks.")
    parser.add_argument("--prepare-only", action="store_true", help="Only generate metadata CSVs, no inference.")
    parser.add_argument(
        "--case-prepared-cache-dir",
        default=None,
        help="Shared checkpoint-independent prepared-input cache for rbd_ddg and antibody_opt.",
    )
    parser.add_argument("--refresh-case-prepared-cache", action="store_true")
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
        "--uniair-data-root",
        "--unair-data-root",
        dest="uniair_data_root",
        default=None,
        help="Root containing unzipped UniAIR datasets. Default: <package>/data/external_uniair.",
    )
    parser.add_argument(
        "--uniair-reuse-foldx-json-dir",
        "--unair-reuse-foldx-json-dir",
        dest="uniair_reuse_foldx_json_dir",
        default=None,
        help=(
            "Reuse one existing row-level FoldX JSON directory for a single UniAIR task. "
            "For both HER2 and TCR-pMHC, prefer --uniair-reuse-foldx-json-root."
        ),
    )
    parser.add_argument(
        "--uniair-reuse-foldx-json-root",
        "--unair-reuse-foldx-json-root",
        dest="uniair_reuse_foldx_json_root",
        default=None,
        help="Root containing <dataset>/sample_json directories for UniAIR FoldX reuse.",
    )
    parser.add_argument("--her2-reuse-foldx-json-dir", default=None)
    parser.add_argument("--tcr-pmhc-reuse-foldx-json-dir", default=None)
    parser.add_argument("--her2-csv", default=None, help="Optional explicit HER2 UniAIR CSV.")
    parser.add_argument("--tcr-pmhc-csv", default=None, help="Optional explicit TCR-pMHC Atlas UniAIR CSV.")
    parser.add_argument("--her2-precomputed-dir", default=None, help="Shared HER2 .pt/ESM sample cache.")
    parser.add_argument("--tcr-pmhc-precomputed-dir", default=None, help="Shared TCR-pMHC .pt/ESM sample cache.")
    parser.add_argument(
        "--uniair-eval-only",
        action="store_true",
        help="Skip UniAIR preparation and evaluate shared --*-precomputed-dir caches directly.",
    )
    parser.add_argument("--uniair-batch-size", "--unair-batch-size", dest="uniair_batch_size", type=int, default=2)
    parser.add_argument("--uniair-num-workers", "--unair-num-workers", dest="uniair_num_workers", type=int, default=0)
    parser.add_argument("--uniair-esm-batch-size", "--unair-esm-batch-size", dest="uniair_esm_batch_size", type=int, default=8)
    parser.add_argument("--uniair-foldx-workers", "--unair-foldx-workers", dest="uniair_foldx_workers", type=int, default=4)
    parser.add_argument("--uniair-refresh-foldx", "--unair-refresh-foldx", dest="uniair_refresh_foldx", action="store_true")
    parser.add_argument("--uniair-refresh-samples", "--unair-refresh-samples", dest="uniair_refresh_samples", action="store_true")
    parser.add_argument("--uniair-refresh-esm", "--unair-refresh-esm", dest="uniair_refresh_esm", action="store_true")
    parser.add_argument("--uniair-allow-zero-energy", "--unair-allow-zero-energy", dest="uniair_allow_zero_energy", action="store_true")
    parser.add_argument("--uniair-skip-wt-aa-check", "--unair-skip-wt-aa-check", dest="uniair_skip_wt_aa_check", action="store_true")
    parser.add_argument("--uniair-strict-tcr-wt-aa-check", action="store_true")
    parser.add_argument(
        "--uniair-skip-mut-pdb-sequence-check",
        "--unair-skip-mut-pdb-sequence-check",
        dest="uniair_skip_mut_pdb_sequence_check",
        action="store_true",
    )
    parser.add_argument("--uniair-limit", "--unair-limit", dest="uniair_limit", type=int, default=None)
    parser.add_argument(
        "--isolated-foldx-cache",
        action="store_true",
        help="Use a separate FOLDX_CACHE_DIR under each task output directory during parallel runs.",
    )
    parser.add_argument("--unsafe-skip-confirmation", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def parse_task_list(raw: str) -> List[str]:
    tasks = [t.strip() for t in raw.split(",") if t.strip()]
    seen = set()
    deduped = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    for t in deduped:
        if t not in ALL_TASKS:
            raise ValueError(f"Unknown task {t!r}. Valid tasks: {', '.join(ALL_TASKS)}")
    return deduped


def maybe_add_flag(cmd: List[str], flag: str, value) -> List[str]:
    if value is None:
        return cmd
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
    else:
        cmd.extend([flag, str(value)])
    return cmd


def abs_path_or_none(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(path)


def dataset_reuse_arg_name(dataset: str) -> str:
    if dataset == "HER2":
        return "her2_reuse_foldx_json_dir"
    if dataset == "TCR_pMHC_Atlas":
        return "tcr_pmhc_reuse_foldx_json_dir"
    raise ValueError(f"Unsupported UniAIR dataset: {dataset}")


def dataset_csv_arg_name(dataset: str) -> str:
    if dataset == "HER2":
        return "her2_csv"
    if dataset == "TCR_pMHC_Atlas":
        return "tcr_pmhc_csv"
    raise ValueError(f"Unsupported UniAIR dataset: {dataset}")


def dataset_precomputed_arg_name(dataset: str) -> str:
    if dataset == "HER2":
        return "her2_precomputed_dir"
    if dataset == "TCR_pMHC_Atlas":
        return "tcr_pmhc_precomputed_dir"
    raise ValueError(f"Unsupported UniAIR dataset: {dataset}")


def resolve_uniair_reuse_dir(dataset: str, args: argparse.Namespace) -> Optional[str]:
    specific = abs_path_or_none(getattr(args, dataset_reuse_arg_name(dataset), None))
    if specific:
        return specific

    root = abs_path_or_none(args.uniair_reuse_foldx_json_root)
    if root:
        return os.path.join(root, dataset, "sample_json")

    generic = abs_path_or_none(args.uniair_reuse_foldx_json_dir)
    if not generic:
        return None

    dataset_sample_json = os.path.join(generic, dataset, "sample_json")
    if os.path.isdir(dataset_sample_json):
        return dataset_sample_json

    dataset_dir = os.path.join(generic, dataset)
    if os.path.isdir(dataset_dir):
        return dataset_dir

    return generic


def validate_uniair_reuse_args(tasks: List[str], args: argparse.Namespace) -> None:
    uniair_tasks = [task for task in tasks if task in UNIAIR_TASKS]
    if len(uniair_tasks) <= 1 or not args.uniair_reuse_foldx_json_dir:
        return

    generic = abs_path_or_none(args.uniair_reuse_foldx_json_dir)
    has_dataset_layout = all(
        os.path.isdir(os.path.join(generic, dataset, "sample_json"))
        or os.path.isdir(os.path.join(generic, dataset))
        for dataset in uniair_tasks
    )
    has_all_specific = all(
        getattr(args, dataset_reuse_arg_name(dataset), None)
        for dataset in uniair_tasks
    )
    if not has_dataset_layout and not has_all_specific:
        raise ValueError(
            "--uniair-reuse-foldx-json-dir is ambiguous when running multiple UniAIR datasets. "
            "Pass --uniair-reuse-foldx-json-root with <root>/<dataset>/sample_json, "
            "or pass --her2-reuse-foldx-json-dir and --tcr-pmhc-reuse-foldx-json-dir."
        )


def describe_uniair_reuse(tasks: List[str], args: argparse.Namespace) -> str:
    parts = []
    for dataset in tasks:
        if dataset not in UNIAIR_TASKS:
            continue
        parts.append(f"{dataset}={resolve_uniair_reuse_dir(dataset, args) or '(compute/write under output)'}")
    return "; ".join(parts)


def build_case_study_cmd(task: str, args: argparse.Namespace, base_out_dir: str) -> List[str]:
    out_dir = os.path.join(base_out_dir, task)
    cmd = [args.python, CASE_STUDY_UNIFIED, task]
    cmd.extend(["--ckpt", args.ckpt])
    cmd.extend(["--out-dir", out_dir])
    if args.skip_plots:
        cmd.append("--skip-plots")
    if args.skip_metrics:
        cmd.append("--skip-metrics")
    if args.prepare_only:
        cmd.append("--prepare-only")
    if args.case_skip_wt_aa_check:
        cmd.append("--skip-wt-aa-check")
    if task == "antibody_opt" and args.case_site_baseline_correction:
        cmd.append("--site-baseline-correction")
    cmd = maybe_add_flag(cmd, "--prepared-cache-dir", abs_path_or_none(args.case_prepared_cache_dir))
    cmd = maybe_add_flag(cmd, "--refresh-prepared-cache", args.refresh_case_prepared_cache)
    return cmd


def build_uniair_cmd(dataset: str, args: argparse.Namespace, base_out_dir: str) -> List[str]:
    out_dir = os.path.join(base_out_dir, dataset)
    reuse_foldx_dir = resolve_uniair_reuse_dir(dataset, args)
    csv_path = abs_path_or_none(getattr(args, dataset_csv_arg_name(dataset), None))
    shared_precomputed_dir = abs_path_or_none(
        getattr(args, dataset_precomputed_arg_name(dataset), None)
    )
    if args.uniair_eval_only:
        if not shared_precomputed_dir:
            raise ValueError(
                f"--uniair-eval-only requires --{dataset_precomputed_arg_name(dataset).replace('_', '-')}."
            )
        return [
            args.python,
            UNIAIR_EVAL,
            "--dataset",
            dataset,
            "--precomputed-dir",
            shared_precomputed_dir,
            "--ckpt",
            args.ckpt,
            "--out-dir",
            os.path.join(out_dir, "eval"),
            "--batch-size",
            str(args.uniair_batch_size),
            "--num-workers",
            str(args.uniair_num_workers),
        ]

    cmd = [args.python, UNIAIR_RUN_PIPELINE]
    cmd.extend(["--dataset", dataset])
    cmd.extend(["--ckpt", args.ckpt])
    cmd.extend([
        "--data-root",
        abs_path_or_none(args.uniair_data_root)
        or os.path.join(config.DATA_DIR, "external_uniair"),
    ])
    cmd.extend(["--metadata-csv", os.path.join(out_dir, f"{dataset}_metadata.csv")])
    cmd.extend(["--precomputed-dir", shared_precomputed_dir or os.path.join(out_dir, "precomputed")])
    cmd.extend(["--out-dir", os.path.join(out_dir, "eval")])
    if reuse_foldx_dir:
        cmd.extend(["--reuse-foldx-json-dir", reuse_foldx_dir])
    else:
        cmd.extend(["--foldx-json-dir", os.path.join(out_dir, "sample_json")])
    cmd = maybe_add_flag(cmd, "--csv", csv_path)
    cmd = maybe_add_flag(cmd, "--batch-size", args.uniair_batch_size)
    cmd = maybe_add_flag(cmd, "--num-workers", args.uniair_num_workers)
    cmd = maybe_add_flag(cmd, "--esm-batch-size", args.uniair_esm_batch_size)
    cmd = maybe_add_flag(cmd, "--foldx-workers", args.uniair_foldx_workers)
    cmd = maybe_add_flag(cmd, "--refresh-foldx", args.uniair_refresh_foldx)
    cmd = maybe_add_flag(cmd, "--refresh-samples", args.uniair_refresh_samples)
    cmd = maybe_add_flag(cmd, "--refresh-esm", args.uniair_refresh_esm)
    cmd = maybe_add_flag(cmd, "--prepare-only", args.prepare_only)
    cmd = maybe_add_flag(cmd, "--allow-zero-energy", args.uniair_allow_zero_energy)
    skip_tcr_wt_check = dataset == "TCR_pMHC_Atlas" and not args.uniair_strict_tcr_wt_aa_check
    cmd = maybe_add_flag(cmd, "--skip-wt-aa-check", args.uniair_skip_wt_aa_check or skip_tcr_wt_check)
    cmd = maybe_add_flag(cmd, "--skip-mut-pdb-sequence-check", args.uniair_skip_mut_pdb_sequence_check)
    cmd = maybe_add_flag(cmd, "--limit", args.uniair_limit)
    return cmd


def build_cmd(task: str, args: argparse.Namespace, base_out_dir: str) -> List[str]:
    if task in CASE_STUDY_TASKS:
        return build_case_study_cmd(task, args, base_out_dir)
    return build_uniair_cmd(task, args, base_out_dir)


def task_log_path(task: str, base_out_dir: str) -> str:
    return os.path.join(base_out_dir, f"{task}.log")


_parallel_lock = threading.Lock()


def _task_env(task: str, gpu_id: str, base_out_dir: str, args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    task_foldx_temp = os.path.join(base_out_dir, task, "foldx_temp")
    os.makedirs(task_foldx_temp, exist_ok=True)
    env["FOLDX_TEMP_DIR"] = task_foldx_temp
    if args.isolated_foldx_cache:
        task_foldx_cache = os.path.join(base_out_dir, task, "foldx_cache")
        os.makedirs(task_foldx_cache, exist_ok=True)
        env["FOLDX_CACHE_DIR"] = task_foldx_cache
    else:
        env.setdefault("FOLDX_CACHE_DIR", config.FOLDX_CACHE_DIR)
    return env


def _stream_stderr(stream, task: str, gpu_id: str, log_path: str, stop_event: threading.Event) -> None:
    COLORS = ["\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
    c = COLORS[int(gpu_id) % len(COLORS)]
    reset = "\033[0m"
    prefix = f"{c}[{task}]{reset} "
    try:
        with open(log_path, "a", encoding="utf-8", errors="replace") as log_fh:
            for line in iter(stream.readline, ""):
                stripped = line.rstrip("\n\r")
                log_fh.write(line)
                log_fh.flush()
                if "\r" in stripped or any(tag in stripped for tag in ("%", "it/s", "s/it", "|", "[", "]")):
                    _print_inline(f"{prefix}{stripped}")
                else:
                    with _parallel_lock:
                        print(f"{prefix}{stripped}", flush=True)
    except Exception:
        pass
    finally:
        stop_event.set()


def _print_inline(text: str) -> None:
    with _parallel_lock:
        sys.stdout.write(f"\r\033[K{text}")
        sys.stdout.flush()


def run_one_parallel(
    task: str,
    gpu_id: str,
    cmd: List[str],
    log_path: str,
    base_out_dir: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    env = _task_env(task, gpu_id, base_out_dir, args)
    short_label = task[:12].ljust(12)

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_fh:
        log_fh.write(f"CUDA_VISIBLE_DEVICES={gpu_id}\ncmd: {' '.join(cmd)}\n---\n")
        log_fh.flush()

    process = subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stop_event = threading.Event()
    t_err = threading.Thread(target=_stream_stderr, args=(process.stderr, task, gpu_id, log_path, stop_event), daemon=True)
    t_out = threading.Thread(target=_stream_stderr, args=(process.stdout, task, gpu_id, log_path, stop_event), daemon=True)
    t_err.start()
    t_out.start()

    t0 = time.time()
    returncode = process.wait()
    t_err.join(timeout=5)
    t_out.join(timeout=5)
    elapsed = time.time() - t0

    result = {"task": task, "gpu": gpu_id, "returncode": returncode, "elapsed": elapsed, "log": os.path.abspath(log_path)}
    with _parallel_lock:
        status = "OK" if returncode == 0 else f"FAIL (rc={returncode})"
        print(f"\n[{status}] {TASK_DISPLAY[task]}  |  GPU {gpu_id}  |  {elapsed:.1f}s  |  log: {log_path}")
    return result


def run_sequential(tasks: List[str], args: argparse.Namespace, base_out_dir: str) -> Dict[str, str]:
    failed: Dict[str, str] = {}
    for i, task in enumerate(tasks, start=1):
        task_t0 = time.time()
        step_label = f"[{i}/{len(tasks)}]"
        cmd = build_cmd(task, args, base_out_dir)
        print(f"\n{step_label}  {TASK_DISPLAY[task]}")
        print(f"  cmd: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
            task_elapsed = time.time() - task_t0
            print(f"{step_label}  {task} finished in {task_elapsed:.1f}s.")
        except subprocess.CalledProcessError as exc:
            failed[task] = str(exc)
            print(f"{step_label}  {task} FAILED: {exc}")
        except Exception as exc:
            failed[task] = str(exc)
            print(f"{step_label}  {task} FAILED: {exc}")
    return failed


def _parse_gpus(gpu_arg: str) -> List[str]:
    return [g.strip() for g in gpu_arg.split(",") if g.strip()]


def run_parallel(tasks: List[str], gpus: List[str], args: argparse.Namespace, base_out_dir: str) -> Dict[str, str]:
    if len(tasks) > len(gpus):
        raise ValueError(
            f"Need at least {len(tasks)} GPUs (got {len(gpus)}: {gpus}). "
            "Provide enough GPU IDs or reduce --tasks."
        )

    os.makedirs(base_out_dir, exist_ok=True)
    futures = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        for task, gpu_id in zip(tasks, gpus):
            cmd = build_cmd(task, args, base_out_dir)
            log_path = task_log_path(task, base_out_dir)
            futures.append(executor.submit(run_one_parallel, task, gpu_id, cmd, log_path, base_out_dir, args))

    failed: Dict[str, str] = {}
    for future in as_completed(futures):
        result = future.result()
        if result["returncode"] != 0:
            failed[result["task"]] = f"returncode={result['returncode']} | log={result['log']}"
    return failed


def main() -> None:
    args_t0 = time.time()
    parsed = parse_args()

    tasks = parse_task_list(parsed.tasks) if parsed.tasks else list(ALL_TASKS)
    gpus = _parse_gpus(parsed.gpus) if parsed.gpus else []
    parsed.ckpt = os.path.abspath(parsed.ckpt)
    ckpt_abs = parsed.ckpt
    base_out_dir = os.path.abspath(
        parsed.out_dir
        or os.path.join(
            config.OUTPUT_DIR,
            "case_studies",
            "all_tasks_beneficial_grouprank_muttype_localtoken32",
        )
    )

    has_uniair = any(t in UNIAIR_TASKS for t in tasks)
    validate_uniair_reuse_args(tasks, parsed)
    foldx_reuse = describe_uniair_reuse(tasks, parsed)

    print("=" * 80)
    print("  RUN ALL CASE STUDIES")
    print(f"  Checkpoint : {ckpt_abs}")
    print(f"  Output dir : {base_out_dir}")
    print(f"  Tasks      : {', '.join(tasks)}")
    if gpus:
        assignment = "  " + "  ".join(f"GPU{g}={t}" for t, g in zip(tasks, gpus))
        print(f"  GPUs       : {assignment}")
    else:
        print("  Mode       : sequential")
    if has_uniair:
        print(f"  FoldX reuse: {foldx_reuse or '(compute fresh)'}")
    print("=" * 80)

    if not parsed.unsafe_skip_confirmation:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    os.makedirs(base_out_dir, exist_ok=True)

    if gpus:
        failed = run_parallel(tasks, gpus, parsed, base_out_dir)
    else:
        failed = run_sequential(tasks, parsed, base_out_dir)

    total_elapsed = time.time() - args_t0
    print("\n" + "=" * 80)
    print(f"  ALL TASKS FINISHED in {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    ok = len(tasks) - len(failed)
    print(f"  OK: {ok}  |  FAILED: {len(failed)}")
    if failed:
        for t, err in failed.items():
            print(f"  FAIL  {t}: {err}")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
