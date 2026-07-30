import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

import config
from case_study_foldx import CaseStudyFoldXBuilder
from case_study_infer import (
    ESMEmbeddingHelper,
    canonicalize_mutation_string,
    infer_one,
    normalize_chain_group,
    parse_mutation_token,
    prepare_model_inputs,
)
from case_study_metrics import ranking_metrics, regression_metrics, write_summary
from case_study_plots import plot_ranking, plot_regression
from foldx_processor import FoldXProcessor
from model import ESM_FoldX_DDAffinity, ESM_RAAD_FoldX_DDAffinity
from run_case_study import (
    CASESTUDY_DIR,
    autodetect_label_col,
    build_metadata_rows,
    build_scan_rows,
    read_mutation_rows,
    resolve_defaults,
    validate_args,
    validate_mutation_strings,
    write_csv,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


CASE_PREPARED_CACHE_VERSION = 7
NOOP_FOLDX_FEATURE_VERSION = 1


def _mutation_default_chain(row: Dict[str, object]) -> Optional[str]:
    mutation_chain = str(row.get("mutation_chain", "")).strip()
    if mutation_chain:
        return mutation_chain
    group1 = normalize_chain_group(str(row.get("chain_group_1", "")))
    return group1[0] if len(group1) == 1 else None


def _site_baseline_spec(row: Dict[str, object]) -> tuple:
    default_chain = _mutation_default_chain(row)
    normalized = canonicalize_mutation_string(
        str(row["mutation"]),
        default_chain=default_chain,
    )
    site_parts = []
    self_tokens = []
    is_noop = True
    for token in normalized.split(","):
        orig_aa, chain_id, residue_id, new_aa = parse_mutation_token(
            token,
            default_chain=default_chain,
        )
        orig_aa = orig_aa.upper()
        new_aa = new_aa.upper()
        site_parts.append(f"{chain_id}:{residue_id}")
        self_tokens.append(f"{orig_aa}{chain_id}{residue_id}{orig_aa}")
        is_noop = is_noop and orig_aa == new_aa
    return ",".join(site_parts), ",".join(self_tokens), is_noop


def build_site_baseline_rows(
    metadata_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    baseline_rows: List[Dict[str, object]] = []
    seen = set()
    for row in metadata_rows:
        site_key, self_mutation, _ = _site_baseline_spec(row)
        if site_key in seen:
            continue
        seen.add(site_key)
        baseline_row = dict(row)
        baseline_row["case_id"] = f"site_baseline:{site_key}"
        baseline_row["mutation"] = self_mutation
        baseline_row["is_favorable"] = ""
        baseline_row["notes"] = "site_self_baseline"
        baseline_row["site_baseline_key"] = site_key
        baseline_rows.append(baseline_row)
    return baseline_rows


def apply_site_baseline_correction(
    prediction_rows: List[Dict[str, object]],
    baseline_prediction_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    baseline_by_site = {
        str(row["site_baseline_key"]): float(row["pred_ddg"])
        for row in baseline_prediction_rows
    }
    corrected_rows = []
    for row in prediction_rows:
        site_key, self_mutation, _ = _site_baseline_spec(row)
        if site_key not in baseline_by_site:
            raise KeyError(f"Missing site baseline prediction for {site_key}.")
        raw_ddg = float(row["pred_ddg"])
        baseline_ddg = baseline_by_site[site_key]
        corrected = dict(row)
        corrected["pred_ddg_raw"] = raw_ddg
        corrected["site_baseline_mutation"] = self_mutation
        corrected["site_baseline_ddg"] = baseline_ddg
        corrected["pred_ddg_corrected"] = raw_ddg - baseline_ddg
        corrected_rows.append(corrected)
    return corrected_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified case-study entry point: prepare metadata, run inference, compute metrics, and generate plots."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    for task_name, help_text in [
        ("rbd_ddg", "Run the 6M0J RBD mutation regression case study."),
        ("antibody_opt", "Run antibody optimization on 7FAE_RBD_Fv with a candidate list or an exhaustive scan."),
    ]:
        sub = subparsers.add_parser(task_name, help=help_text)
        sub.add_argument(
            "--ckpt",
            default=config.BEST_MODEL_PATH,
            help="Checkpoint path. Default: best_model.pth.",
        )
        sub.add_argument("--out-dir", default=None, help="Output directory. Defaults to outputs/<seed>/case_studies/<task>.")
        sub.add_argument("--mutation-csv", default=None, help="Mutation table in CSV/YAML format.")
        sub.add_argument("--mutation-col", default="mutation", help="Mutation column name.")
        sub.add_argument("--label-col", default=None, help="Optional regression label column.")
        sub.add_argument("--favorable-col", default="is_favorable", help="Optional ranking label column.")
        sub.add_argument("--prepare-only", action="store_true", help="Only generate metadata CSV.")
        sub.add_argument(
            "--skip-wt-aa-check",
            action="store_true",
            help="Relax WT amino-acid checks when preparing mutation-aware structure features.",
        )
        sub.add_argument(
            "--prepared-cache-dir",
            default=getattr(config, "CASE_STUDY_PREPARED_CACHE_DIR", None),
            help="Checkpoint-independent cache for FoldX, structure, and ESM model inputs.",
        )
        sub.add_argument(
            "--refresh-prepared-cache",
            action="store_true",
            help="Recompute checkpoint-independent prepared inputs instead of reusing the cache.",
        )
        sub.add_argument("--skip-metrics", action="store_true", help="Skip metric computation.")
        sub.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
        sub.add_argument("--pdb-path", default=None, help="Override the preset PDB path.")
        sub.add_argument("--chain-group-1", default=None, help="Override partner-1 chain group.")
        sub.add_argument("--chain-group-2", default=None, help="Override partner-2 chain group.")
        sub.add_argument("--scan-positions", default=None, help="For antibody_opt only. Comma-separated positions.")
        sub.add_argument("--candidate-aas", default="ACDEFGHIKLMNPQRSTVWY", help="Amino acids used for exhaustive scanning.")
        sub.add_argument("--topk", type=int, nargs="*", default=[10, 20, 50, 100], help="Top-k cutoffs for ranking metrics.")
        sub.add_argument("--drop-zero-labels", action="store_true", help="For regression only. Exclude rows whose experimental_ddg is exactly zero.")
        sub.add_argument(
            "--site-baseline-correction",
            action="store_true",
            help=(
                "For antibody_opt, infer one self mutation per site and rank by "
                "pred_ddg_corrected = pred_ddg - site_baseline_ddg."
            ),
        )

    args = parser.parse_args()
    validate_args(args)
    return args


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


def _tree_to_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return value


def _tree_to_device(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _tree_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_device(item, device) for item in value)
    return value


def _prepared_cache_identity(
    row: Dict[str, object],
    strict_wt_check: bool = True,
) -> Dict[str, object]:
    pdb_path = os.path.abspath(str(row["pdb_path"]))
    try:
        stat = os.stat(pdb_path)
        pdb_size = int(stat.st_size)
        pdb_mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        pdb_size = -1
        pdb_mtime_ns = -1

    identity = {
        "version": CASE_PREPARED_CACHE_VERSION,
        "pdb_path": pdb_path,
        "pdb_size": pdb_size,
        "pdb_mtime_ns": pdb_mtime_ns,
        "chain_group_1": str(row["chain_group_1"]),
        "chain_group_2": str(row["chain_group_2"]),
        "mutation": str(row["mutation"]),
        "mutation_chain": str(row.get("mutation_chain", "")),
        "esm_model_name": str(config.ESM_MODEL_NAME),
        "use_precomputed_esm": bool(getattr(config, "USE_PRECOMPUTED_ESM", False)),
        "esm_mutation_window_radius": int(getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8)),
        "esm_local_max_tokens": int(getattr(config, "ESM_LOCAL_MAX_TOKENS", 32)),
        "esm_local_token_version": int(getattr(config, "ESM_LOCAL_TOKEN_VERSION", 1)),
        "struct_local_max_residues": int(
            getattr(config, "STRUCT_LOCAL_MAX_RESIDUES", 32)
        ),
        "foldx_feature_mode": str(getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta")),
        "foldx_version": str(getattr(config, "FOLDX_VERSION", "foldx")),
        "foldx_path": os.path.abspath(config.FOLDX_PATH),
        "mutation_local_radius": float(getattr(config, "MUTATION_LOCAL_RADIUS", 10.0)),
        "max_structure_atoms": int(getattr(config, "MAX_STRUCTURE_ATOMS", 4096)),
        "mutation_type_feature_version": int(
            getattr(config, "MUTATION_TYPE_FEATURE_VERSION", 2)
        ),
        "strict_wt_check": bool(strict_wt_check),
    }
    _, _, is_noop = _site_baseline_spec(row)
    if is_noop:
        identity["noop_foldx_feature_version"] = NOOP_FOLDX_FEATURE_VERSION
    return identity


def _prepared_cache_path(
    row: Dict[str, object],
    cache_dir: Optional[str],
    strict_wt_check: bool = True,
) -> Optional[str]:
    if not cache_dir:
        return None
    identity = _prepared_cache_identity(
        row,
        strict_wt_check=strict_wt_check,
    )
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return os.path.join(os.path.abspath(cache_dir), f"{digest}.pt")


def _load_prepared_sample(
    row: Dict[str, object],
    cache_dir: Optional[str],
    device,
    refresh: bool,
    strict_wt_check: bool = True,
):
    import torch

    cache_path = _prepared_cache_path(
        row,
        cache_dir,
        strict_wt_check=strict_wt_check,
    )
    if refresh or not cache_path or not os.path.isfile(cache_path):
        return None, cache_path
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("identity") != _prepared_cache_identity(
            row,
            strict_wt_check=strict_wt_check,
        ):
            return None, cache_path
        sample = payload.get("sample")
        if not isinstance(sample, dict):
            return None, cache_path
        return _tree_to_device(sample, device), cache_path
    except Exception as exc:
        print(f"Prepared cache miss ({cache_path}): {exc}")
        return None, cache_path


def _save_prepared_sample(
    row: Dict[str, object],
    sample: Dict[str, object],
    cache_path: Optional[str],
    strict_wt_check: bool = True,
) -> None:
    import torch

    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "identity": _prepared_cache_identity(
            row,
            strict_wt_check=strict_wt_check,
        ),
        "sample": _tree_to_cpu(sample),
    }
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


def load_weights(model, ckpt_path: str, device):
    checkpoint = __import__("torch").load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {ckpt_path}")

    load_errors = []
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as raw_error:
        load_errors.append(f"raw state_dict failed: {raw_error}")
        stripped_state = OrderedDict()
        prefixed_state = OrderedDict()
        for key, value in checkpoint.items():
            stripped_state[key.replace("module.", "", 1)] = value
            prefixed_state[key if key.startswith("module.") else f"module.{key}"] = value

        try:
            model.load_state_dict(stripped_state)
        except RuntimeError as stripped_error:
            load_errors.append(
                f"module-prefix stripped state_dict failed: {stripped_error}"
            )
            try:
                model.load_state_dict(prefixed_state)
            except RuntimeError as prefixed_error:
                load_errors.append(
                    f"module-prefix added state_dict failed: {prefixed_error}"
                )
                raise RuntimeError(
                    "Checkpoint is incompatible with the local-token32 model. "
                    "Partial state-dict loading is intentionally disabled. "
                    "Attempts: " + " | ".join(load_errors)
                ) from prefixed_error

    model.to(device)
    model.eval()
    return model


def resolve_checkpoint_path(ckpt_path: str) -> str:
    ckpt_path = os.path.expanduser(str(ckpt_path).strip())
    if not ckpt_path:
        raise ValueError("Checkpoint path is empty.")

    candidates = (
        [os.path.abspath(ckpt_path)]
        if os.path.isabs(ckpt_path)
        else [
            os.path.abspath(ckpt_path),
            os.path.abspath(os.path.join(SCRIPT_DIR, ckpt_path)),
            os.path.abspath(os.path.join(config.BASE_DIR, ckpt_path)),
        ]
    )
    tried = []
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Checkpoint was not found: {ckpt_path}. Tried: {tried}"
    )


def run_inference_rows(
    metadata_rows: List[Dict[str, object]],
    ckpt_path: str,
    prepared_cache_dir: Optional[str] = None,
    refresh_prepared_cache: bool = False,
    strict_wt_check: bool = True,
) -> List[Dict[str, object]]:
    import torch

    device = config.DEVICE
    ckpt_path = resolve_checkpoint_path(ckpt_path)

    model = build_model(use_precomputed_esm=getattr(config, "USE_PRECOMPUTED_ESM", False))
    model = load_weights(model, ckpt_path, device)

    foldx_processor = FoldXProcessor(
        foldx_path=config.FOLDX_PATH,
        temp_dir=config.FOLDX_TEMP_DIR,
        cache_dir=config.FOLDX_CACHE_DIR,
    )
    foldx_builder = CaseStudyFoldXBuilder(
        foldx_path=config.FOLDX_PATH,
        temp_dir=os.path.join(config.FOLDX_TEMP_DIR, "case_study"),
        cache_dir=os.path.join(config.FOLDX_CACHE_DIR, "case_study_mutants"),
    )

    esm_helper = None

    output_rows: List[Dict[str, object]] = []
    cache_hits = 0
    cache_misses = 0
    for row in tqdm(metadata_rows, desc="Running case study", unit="case"):
        sample, cache_path = _load_prepared_sample(
            row,
            cache_dir=prepared_cache_dir,
            device=device,
            refresh=refresh_prepared_cache,
            strict_wt_check=strict_wt_check,
        )
        if sample is None:
            cache_misses += 1
            if esm_helper is None and getattr(config, "USE_PRECOMPUTED_ESM", False):
                esm_helper = ESMEmbeddingHelper(config.ESM_MODEL_NAME, device)
            try:
                sample = prepare_model_inputs(
                    dict(row),
                    foldx_processor,
                    foldx_builder,
                    esm_helper,
                    device,
                    strict_wt_check=strict_wt_check,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to prepare case-study sample "
                    f"case_id={row.get('case_id', '')}, "
                    f"mutation={row.get('mutation', '')}, "
                    f"pdb_path={row.get('pdb_path', '')}"
                ) from exc
            _save_prepared_sample(
                row,
                sample,
                cache_path,
                strict_wt_check=strict_wt_check,
            )
        else:
            cache_hits += 1
        pred_ddg = infer_one(model, sample)
        out_row = dict(row)
        out_row["pred_ddg"] = pred_ddg
        out_row["foldx_energy"] = sample["foldx_energy"]
        out_row["foldx_mut_energy"] = sample.get("foldx_mut_energy", "")
        out_row["foldx_delta_interaction"] = sample.get("foldx_delta_interaction", "")
        out_row["foldx_buildmodel_ddg"] = sample.get("foldx_buildmodel_ddg", "")
        out_row["foldx_feature_mode"] = sample.get("foldx_feature_mode", "")
        out_row["foldx_version"] = sample.get("foldx_version", "")
        out_row["foldx_path"] = sample.get("foldx_path", "")
        out_row["normalized_mutation"] = sample["normalized_mutation"]
        out_row["structure_pdb_path"] = sample.get("structure_pdb_path", "")
        out_row["mutant_pdb_path"] = sample["mutant_pdb_path"]
        output_rows.append(out_row)

    if prepared_cache_dir:
        print(
            f"Prepared input cache: hits={cache_hits}, misses={cache_misses}, "
            f"dir={os.path.abspath(prepared_cache_dir)}"
        )
    return output_rows


def main() -> None:
    args = parse_args()
    if args.site_baseline_correction and args.task != "antibody_opt":
        raise ValueError("--site-baseline-correction is only valid for antibody_opt.")
    defaults = resolve_defaults(args)

    pdb_path = os.path.abspath(args.pdb_path or defaults["pdb_path"])
    chain_group_1 = args.chain_group_1 or defaults["chain_group_1"]
    chain_group_2 = args.chain_group_2 or defaults["chain_group_2"]
    out_dir = os.path.abspath(
        args.out_dir
        or os.path.join(config.OUTPUT_DIR, "case_studies", args.task)
    )
    os.makedirs(out_dir, exist_ok=True)

    if args.scan_positions:
        source_rows = build_scan_rows(pdb_path, positions=args.scan_positions.split(","), candidate_aas=args.candidate_aas)
    else:
        mutation_csv = os.path.abspath(args.mutation_csv or defaults["mutation_csv"])
        source_rows = read_mutation_rows(mutation_csv, candidate_aas=args.candidate_aas)

    if not source_rows:
        raise ValueError("No input rows were found for the case study.")

    label_col = args.label_col or autodetect_label_col(source_rows)
    favorable_col = args.favorable_col if args.task == "antibody_opt" else None
    validate_mutation_strings(source_rows, args.mutation_col, chain_group_1)

    metadata_rows = build_metadata_rows(
        source_rows=source_rows,
        task=args.task,
        pdb_path=pdb_path,
        chain_group_1=chain_group_1,
        chain_group_2=chain_group_2,
        mutation_col=args.mutation_col,
        label_col=label_col,
        favorable_col=favorable_col,
    )

    metadata_csv = os.path.join(out_dir, f"{args.task}_metadata.csv")
    predictions_csv = os.path.join(out_dir, f"{args.task}_predictions.csv")
    baselines_csv = os.path.join(out_dir, f"{args.task}_site_baselines.csv")
    summary_csv = os.path.join(out_dir, f"{args.task}_summary.csv")
    plots_dir = os.path.join(out_dir, "plots")

    write_csv(metadata_csv, metadata_rows)
    print(f"Prepared metadata: {metadata_csv}")

    if args.prepare_only:
        return

    baseline_metadata_rows = (
        build_site_baseline_rows(metadata_rows)
        if args.site_baseline_correction
        else []
    )
    inference_rows = metadata_rows + baseline_metadata_rows
    all_prediction_rows = run_inference_rows(
        inference_rows,
        args.ckpt,
        prepared_cache_dir=args.prepared_cache_dir,
        refresh_prepared_cache=args.refresh_prepared_cache,
        strict_wt_check=not args.skip_wt_aa_check,
    )
    prediction_rows = all_prediction_rows[: len(metadata_rows)]
    ranking_score_col = "pred_ddg"
    if baseline_metadata_rows:
        baseline_prediction_rows = all_prediction_rows[len(metadata_rows) :]
        prediction_rows = apply_site_baseline_correction(
            prediction_rows,
            baseline_prediction_rows,
        )
        write_csv(baselines_csv, baseline_prediction_rows)
        print(f"Site baselines saved to: {baselines_csv}")
        ranking_score_col = "pred_ddg_corrected"
    write_csv(predictions_csv, prediction_rows)
    print(f"Predictions saved to: {predictions_csv}")

    metrics_task = "regression" if args.task == "rbd_ddg" else "ranking"
    has_regression_labels = any(row.get("experimental_ddg", "") != "" for row in metadata_rows)
    has_ranking_labels = any(row.get("is_favorable", "") != "" for row in metadata_rows)

    if not args.skip_metrics:
        should_run_metrics = (
            metrics_task == "regression" and has_regression_labels
        ) or (
            metrics_task == "ranking" and has_ranking_labels
        )
        if should_run_metrics:
            if metrics_task == "regression":
                metric_rows = [
                    regression_metrics(
                        prediction_rows,
                        drop_zero_labels=args.drop_zero_labels,
                    )
                ]
            else:
                metric_rows = [
                    ranking_metrics(
                        prediction_rows,
                        args.topk,
                        score_col=ranking_score_col,
                    )
                ]
                if ranking_score_col != "pred_ddg":
                    metric_rows.append(
                        ranking_metrics(
                            prediction_rows,
                            args.topk,
                            score_col="pred_ddg",
                        )
                    )
            write_summary(summary_csv, metric_rows)
            print(f"Summary saved to: {summary_csv}")
        else:
            print("Skipping metrics because the required labels are missing.")

    if not args.skip_plots:
        os.makedirs(plots_dir, exist_ok=True)
        if metrics_task == "regression":
            plot_regression(prediction_rows, plots_dir)
        else:
            plot_ranking(
                prediction_rows,
                plots_dir,
                score_col=ranking_score_col,
            )
        print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
