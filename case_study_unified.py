import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

import config
from case_study_foldx import CaseStudyFoldXBuilder
from case_study_infer import (
    ESMEmbeddingHelper,
    infer_one,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified case-study entry point: prepare metadata, run inference, compute metrics, and generate plots."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    for task_name, help_text in [
        ("rbd_ddg", "Run the bundled 6M0J RBD mutation regression case study."),
        ("antibody_opt", "Run antibody optimization on 7FAE_RBD_Fv with a candidate list or an exhaustive scan."),
    ]:
        sub = subparsers.add_parser(task_name, help=help_text)
        sub.add_argument("--ckpt", default="best_model1.pth", help="Checkpoint path.")
        sub.add_argument("--out-dir", default=None, help="Output directory. Defaults to casestudy/results/<task>.")
        sub.add_argument("--mutation-csv", default=None, help="Mutation table in CSV/YAML format.")
        sub.add_argument("--mutation-col", default="mutation", help="Mutation column name.")
        sub.add_argument("--label-col", default=None, help="Optional regression label column.")
        sub.add_argument("--favorable-col", default="is_favorable", help="Optional ranking label column.")
        sub.add_argument("--prepare-only", action="store_true", help="Only generate metadata CSV.")
        sub.add_argument("--skip-metrics", action="store_true", help="Skip metric computation.")
        sub.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
        sub.add_argument("--pdb-path", default=None, help="Override the preset PDB path.")
        sub.add_argument("--chain-group-1", default=None, help="Override partner-1 chain group.")
        sub.add_argument("--chain-group-2", default=None, help="Override partner-2 chain group.")
        sub.add_argument("--scan-positions", default=None, help="For antibody_opt only. Comma-separated positions.")
        sub.add_argument("--candidate-aas", default="ACDEFGHIKLMNPQRSTVWY", help="Amino acids used for exhaustive scanning.")
        sub.add_argument("--topk", type=int, nargs="*", default=[10, 20, 50, 100], help="Top-k cutoffs for ranking metrics.")
        sub.add_argument("--drop-zero-labels", action="store_true", help="For regression only. Exclude rows whose experimental_ddg is exactly zero.")

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
        )

    return ESM_FoldX_DDAffinity(
        esm_model_name=config.ESM_MODEL_NAME,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        use_precomputed_esm=use_precomputed_esm,
    )


def load_weights(model, ckpt_path: str, device):
    checkpoint = __import__("torch").load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        stripped_state = OrderedDict()
        prefixed_state = OrderedDict()
        for key, value in checkpoint.items():
            stripped_state[key.replace("module.", "", 1)] = value
            prefixed_state[key if key.startswith("module.") else f"module.{key}"] = value

        try:
            model.load_state_dict(stripped_state)
        except RuntimeError:
            try:
                model.load_state_dict(prefixed_state)
            except RuntimeError:
                compatible_state = prefixed_state if any(k.startswith("module.") for k in model.state_dict().keys()) else stripped_state
                model.load_state_dict(compatible_state, strict=False)

    model.to(device)
    model.eval()
    return model


def run_inference_rows(metadata_rows: List[Dict[str, object]], ckpt_path: str) -> List[Dict[str, object]]:
    import torch

    device = config.DEVICE
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config.BASE_DIR, ckpt_path)

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
    if getattr(config, "USE_PRECOMPUTED_ESM", False):
        esm_helper = ESMEmbeddingHelper(config.ESM_MODEL_NAME, device)

    output_rows: List[Dict[str, object]] = []
    for row in tqdm(metadata_rows, desc="Running case study", unit="case"):
        sample = prepare_model_inputs(dict(row), foldx_processor, foldx_builder, esm_helper, device)
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

    return output_rows


def main() -> None:
    args = parse_args()
    defaults = resolve_defaults(args)

    pdb_path = os.path.abspath(args.pdb_path or defaults["pdb_path"])
    chain_group_1 = args.chain_group_1 or defaults["chain_group_1"]
    chain_group_2 = args.chain_group_2 or defaults["chain_group_2"]
    out_dir = os.path.abspath(args.out_dir or os.path.join(CASESTUDY_DIR, "results", args.task))
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
    summary_csv = os.path.join(out_dir, f"{args.task}_summary.csv")
    plots_dir = os.path.join(out_dir, "plots")

    write_csv(metadata_csv, metadata_rows)
    print(f"Prepared metadata: {metadata_csv}")

    if args.prepare_only:
        return

    prediction_rows = run_inference_rows(metadata_rows, args.ckpt)
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
                metrics = regression_metrics(prediction_rows, drop_zero_labels=args.drop_zero_labels)
            else:
                metrics = ranking_metrics(prediction_rows, args.topk)
            write_summary(summary_csv, [metrics])
            print(f"Summary saved to: {summary_csv}")
        else:
            print("Skipping metrics because the required labels are missing.")

    if not args.skip_plots:
        os.makedirs(plots_dir, exist_ok=True)
        if metrics_task == "regression":
            plot_regression(prediction_rows, plots_dir)
        else:
            plot_ranking(prediction_rows, plots_dir)
        print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
