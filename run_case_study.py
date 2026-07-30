import argparse
import csv
import os
import subprocess
import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

try:
    import yaml
except ImportError:
    yaml = None

import config

BASE_DIR = config.BASE_DIR
CASESTUDY_DIR = config.CASE_STUDY_DIR
DDAFFINITY_DATA_DIR = config.DATA_DIR
CASE_STUDY_INFER = os.path.join(SCRIPT_DIR, "case_study_infer.py")
CASE_STUDY_METRICS = os.path.join(SCRIPT_DIR, "case_study_metrics.py")
CASE_STUDY_PLOTS = os.path.join(SCRIPT_DIR, "case_study_plots.py")
DEFAULT_AAS = "ACDEFGHIKLMNPQRSTVWY"

THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and run DiffAffinity-style case studies with this local model."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    add_common_args(subparsers.add_parser(
        "rbd_ddg",
        help="Run the 6M0J RBD mutation regression case study.",
    ))
    add_common_args(subparsers.add_parser(
        "antibody_opt",
        help="Run antibody optimization on 7FAE_RBD_Fv with a candidate list or an exhaustive scan.",
    ))

    args = parser.parse_args()
    validate_args(args)
    return args


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ckpt",
        default=config.BEST_MODEL_PATH,
        help="Checkpoint path passed to case_study_infer.py. Default filename: best_model.pth.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to outputs/<seed>/case_studies/<task>.",
    )
    parser.add_argument(
        "--mutation-csv",
        default=None,
        help="Mutation table in CSV/YAML format. For rbd_ddg it defaults to CASE_STUDY_DIR/DDG_6m0j.csv.",
    )
    parser.add_argument(
        "--mutation-col",
        default="mutation",
        help="Column containing mutation strings such as GE339A or TH31Y.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Optional regression label column. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--favorable-col",
        default="is_favorable",
        help="Optional ranking label column for antibody optimization.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate the metadata CSV, without running inference.",
    )
    parser.add_argument(
        "--skip-wt-aa-check",
        action="store_true",
        help="Relax WT amino-acid checks when preparing mutation-aware structure features.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip case_study_metrics.py even if labels are available.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip case_study_plots.py.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the helper scripts.",
    )
    parser.add_argument(
        "--pdb-path",
        default=None,
        help="Override the preset PDB path.",
    )
    parser.add_argument(
        "--chain-group-1",
        default=None,
        help="Override partner-1 chain group. Examples: E or HL.",
    )
    parser.add_argument(
        "--chain-group-2",
        default=None,
        help="Override partner-2 chain group.",
    )
    parser.add_argument(
        "--scan-positions",
        default=None,
        help="For antibody_opt only. Comma-separated positions like H31,H52,L32.",
    )
    parser.add_argument(
        "--candidate-aas",
        default=DEFAULT_AAS,
        help="Amino acids used for exhaustive scanning. Wild-type residues are skipped automatically.",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.task == "rbd_ddg" and args.scan_positions:
        raise ValueError("--scan-positions is only supported for antibody_opt.")
    if args.task == "antibody_opt" and not args.scan_positions and args.mutation_csv == "":
        raise ValueError("antibody_opt needs either --mutation-csv or --scan-positions.")


def resolve_defaults(args: argparse.Namespace) -> Dict[str, str]:
    if args.task == "rbd_ddg":
        return {
            "pdb_path": os.path.join(CASESTUDY_DIR, "6M0J.pdb"),
            "chain_group_1": "E",
            "chain_group_2": "A",
            "mutation_csv": os.path.join(CASESTUDY_DIR, "DDG_6m0j.csv"),
        }
    return {
        "pdb_path": os.path.join(CASESTUDY_DIR, "7FAE_RBD_Fv.pdb"),
        "chain_group_1": "HL",
        "chain_group_2": "A",
        "mutation_csv": os.path.join(DDAFFINITY_DATA_DIR, "7FAE_RBD_Fv_mutation.yml"),
    }


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expand_mutation_token(token: str, candidate_aas: str = DEFAULT_AAS) -> List[str]:
    token = token.strip().upper()
    if not token:
        return []
    if "*" not in token:
        return [token]
    if token.count("*") != 1 or not token.endswith("*") or len(token) < 4:
        raise ValueError(f"Unsupported wildcard mutation format: {token!r}")

    wild_type = token[0]
    expanded = []
    for aa in candidate_aas.upper():
        if aa == wild_type:
            continue
        if aa not in expanded:
            expanded.append(f"{token[:-1]}{aa}")
    return expanded


def parse_simple_yaml_lists(path: str) -> Dict[str, List[str]]:
    data: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.endswith(":") and not stripped.startswith("- "):
                current_key = stripped[:-1].strip()
                data.setdefault(current_key, [])
                continue
            if stripped.startswith("- ") and current_key:
                data[current_key].append(stripped[2:].strip())
    return data


def read_yaml_rows(path: str, candidate_aas: str = DEFAULT_AAS) -> List[Dict[str, str]]:
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    else:
        payload = parse_simple_yaml_lists(path)

    mutation_patterns = payload.get("mutations") or []
    favorable_mutations = {
        str(item).strip().upper()
        for item in (payload.get("interest") or [])
        if str(item).strip()
    }

    rows: List[Dict[str, str]] = []
    for pattern in mutation_patterns:
        pattern_text = str(pattern).strip().upper()
        if not pattern_text:
            continue
        for mutation in expand_mutation_token(pattern_text, candidate_aas=candidate_aas):
            rows.append(
                {
                    "mutation": mutation,
                    "is_favorable": "1" if mutation in favorable_mutations else "0",
                    "source_pattern": pattern_text,
                    "notes": "yaml_enumeration",
                }
            )
    return rows


def read_mutation_rows(path: str, candidate_aas: str = DEFAULT_AAS) -> List[Dict[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yml", ".yaml"}:
        return read_yaml_rows(path, candidate_aas=candidate_aas)
    return read_csv_rows(path)


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def autodetect_label_col(rows: Sequence[Dict[str, str]]) -> Optional[str]:
    candidates = [
        "experimental_ddg",
        "delta_bind",
        "ddg",
        "label",
        "y",
    ]
    for name in candidates:
        if rows and name in rows[0]:
            return name
    return None


def mutation_has_explicit_chain(token: str) -> bool:
    token = token.strip()
    return len(token) >= 4 and token[1].isalpha()


def validate_mutation_strings(rows: Sequence[Dict[str, str]], mutation_col: str, chain_group_1: str) -> None:
    if len(chain_group_1) <= 1:
        return

    for idx, row in enumerate(rows, start=1):
        mutation_value = (row.get(mutation_col) or "").strip()
        if not mutation_value:
            raise ValueError(f"Row {idx} is missing the mutation value in column '{mutation_col}'.")
        for token in mutation_value.split(","):
            if not mutation_has_explicit_chain(token):
                raise ValueError(
                    "Multi-chain partner-1 inputs need chain-qualified mutations such as TH31Y or SL32W. "
                    f"Invalid token on row {idx}: {token!r}"
                )


def load_chain_residues(pdb_path: str) -> Dict[str, "OrderedDict[str, str]"]:
    chains: Dict[str, "OrderedDict[str, str]"] = {}
    seen = set()
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM") or len(line) < 27:
                continue
            chain_id = line[21].strip() or "_"
            residue_id = line[22:27].strip()
            residue_name = line[17:20].strip()
            if residue_name not in THREE_TO_ONE:
                continue
            key = (chain_id, residue_id)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(chain_id, OrderedDict())[residue_id] = THREE_TO_ONE[residue_name]
    return chains


def build_scan_rows(
    pdb_path: str,
    positions: Iterable[str],
    candidate_aas: str,
) -> List[Dict[str, str]]:
    chains = load_chain_residues(pdb_path)
    unique_aas = []
    for aa in candidate_aas.upper():
        if aa not in unique_aas:
            unique_aas.append(aa)

    rows: List[Dict[str, str]] = []
    for position in positions:
        position = position.strip()
        if len(position) < 2:
            raise ValueError(f"Invalid position specifier: {position!r}")
        chain_id = position[0]
        residue_id = position[1:]
        if chain_id not in chains:
            raise KeyError(f"Chain {chain_id!r} was not found in {pdb_path}.")
        if residue_id not in chains[chain_id]:
            raise KeyError(f"Residue {chain_id}{residue_id} was not found in {pdb_path}.")
        wild_type = chains[chain_id][residue_id]
        for mutant in unique_aas:
            if mutant == wild_type:
                continue
            rows.append(
                {
                    "mutation": f"{wild_type}{chain_id}{residue_id}{mutant}",
                    "is_favorable": "",
                    "notes": f"exhaustive_scan_{chain_id}{residue_id}",
                }
            )
    return rows


def build_metadata_rows(
    source_rows: Sequence[Dict[str, str]],
    task: str,
    pdb_path: str,
    chain_group_1: str,
    chain_group_2: str,
    mutation_col: str,
    label_col: Optional[str],
    favorable_col: Optional[str],
) -> List[Dict[str, object]]:
    metadata_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(source_rows, start=1):
        mutation_value = (row.get(mutation_col) or "").strip()
        if not mutation_value:
            raise ValueError(f"Row {idx} is missing a mutation in column '{mutation_col}'.")

        metadata = {
            "case_id": idx,
            "pdb_path": os.path.abspath(pdb_path),
            "chain_group_1": chain_group_1,
            "chain_group_2": chain_group_2,
            "mutation": mutation_value,
        }

        for key, value in row.items():
            if key == mutation_col:
                continue
            if value is None or value == "":
                continue
            metadata[key] = value

        if label_col and row.get(label_col, "") != "":
            raw_label = row[label_col]
            if task == "rbd_ddg" and label_col == "delta_bind":
                # The bundled 6M0J case-study table follows binding-effect scores,
                # while this project predicts SKEMPI-style ddG.
                metadata["experimental_ddg"] = -float(raw_label)
                metadata["label_transform"] = "experimental_ddg = -delta_bind"
            else:
                metadata["experimental_ddg"] = raw_label
        if favorable_col and row.get(favorable_col, "") != "":
            metadata["is_favorable"] = row[favorable_col]

        metadata_rows.append(metadata)
    return metadata_rows


def run_command(cmd: Sequence[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def main() -> None:
    args = parse_args()
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
        source_rows = build_scan_rows(
            pdb_path,
            positions=args.scan_positions.split(","),
            candidate_aas=args.candidate_aas,
        )
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

    infer_cmd = [
        args.python,
        CASE_STUDY_INFER,
        "--case-csv",
        metadata_csv,
        "--ckpt",
        args.ckpt,
        "--out-csv",
        predictions_csv,
        "--task",
        args.task,
    ]
    if args.skip_wt_aa_check:
        infer_cmd.append("--skip-wt-aa-check")
    run_command(infer_cmd)

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
            metrics_cmd = [
                args.python,
                CASE_STUDY_METRICS,
                "--pred-csv",
                predictions_csv,
                "--task",
                metrics_task,
                "--summary-out",
                summary_csv,
            ]
            run_command(metrics_cmd)
        else:
            print("Skipping metrics because the required labels are missing.")

    if not args.skip_plots:
        plot_cmd = [
            args.python,
            CASE_STUDY_PLOTS,
            "--pred-csv",
            predictions_csv,
            "--task",
            metrics_task,
            "--out-dir",
            plots_dir,
        ]
        run_command(plot_cmd)

    print(f"Predictions saved to: {predictions_csv}")
    if os.path.exists(summary_csv):
        print(f"Summary saved to: {summary_csv}")
    if os.path.isdir(plots_dir):
        print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
