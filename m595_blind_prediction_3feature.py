import argparse
import hashlib
import json
import os
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for module_path in (PROJECT_ROOT, SCRIPT_DIR):
    if module_path in sys.path:
        sys.path.remove(module_path)
    sys.path.insert(0, module_path)

import numpy as np
import pandas as pd
import torch
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import config
from case_study_infer import ESMEmbeddingHelper
from data_loader import attach_mutation_masks, cached_structure_extraction
from foldx_processor import FoldXProcessor
from model import ESM_FoldX_DDAffinity, ESM_RAAD_FoldX_DDAffinity

try:
    import yaml
except ImportError:
    yaml = None


BASE_DIR = PROJECT_ROOT
ORIGINAL_DDAFFINITY_ROOT = os.path.join(BASE_DIR, "re", "DDAffinity-master")
M595_DATA_ROOT = os.path.join(BASE_DIR, "m595", "M595_cache")
DDAFFINITY_DATA_ROOT = os.path.join(BASE_DIR, "data", "M595_cache")
LEGACY_M595_DATA_ROOT = os.path.join(
    ORIGINAL_DDAFFINITY_ROOT, "data", "SKEMPI2", "M595_cache"
)
DEFAULT_CASE_CONFIG = os.path.join(
    ORIGINAL_DDAFFINITY_ROOT, "configs", "inference", "blind_testing.yml"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    BASE_DIR, "casestudy", "results", "m595_3feature"
)
DEFAULT_M595_FOLDX_CACHE_DIR = os.path.join(BASE_DIR, "m595", "M595_foldx_cache")
M595_FOLDX_CACHE_VERSION = "m595_foldx_cache_v1"
FOLDX_FEATURE_MODE = getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta")
_FOLDX_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run M595 multiple-mutation blind prediction with this folder's "
            "3-feature FoldX checkpoint(s), using the original M595 cache entries."
        )
    )
    parser.add_argument(
        "--case-config",
        default=None,
        help="Original DDAffinity blind-testing YAML used for default M595 paths.",
    )
    parser.add_argument(
        "--ckpt",
        nargs="+",
        default=[config.BEST_MODEL_PATH],
        help=(
            "One or more 3-feature local-model checkpoints. If multiple are provided, the script "
            "can aggregate them as an ensemble."
        ),
    )
    parser.add_argument(
        "--pdb-wt-dir",
        default=None,
        help="Override the wildtype structure directory.",
    )
    parser.add_argument(
        "--pdb-mt-dir",
        default=None,
        help="Override the mutant structure directory.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override the entries cache directory that contains entries.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used for metadata, raw predictions, aggregated predictions, and metrics.",
    )
    parser.add_argument(
        "--device",
        default=str(config.DEVICE),
        help="Torch device string.",
    )
    parser.add_argument(
        "--structure-source",
        choices=["wt", "mt"],
        default="wt",
        help="Which structure to feed into RAAD/FoldX feature extraction.",
    )
    parser.add_argument(
        "--aggregate-mode",
        choices=["auto", "identity", "mean", "ddaffinity"],
        default="auto",
        help=(
            "How to combine multiple checkpoint predictions per mutation. "
            "'ddaffinity' uses -(max-min)/2, matching the original M595 aggregation."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many best-ranked mutations to print.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip three-mode metric computation even if ddG labels exist.",
    )
    parser.add_argument(
        "--foldx-cache-dir",
        default=DEFAULT_M595_FOLDX_CACHE_DIR,
        help="Directory used to store reusable M595 WT/MT FoldX energy JSON files.",
    )
    parser.add_argument(
        "--foldx-workers",
        type=int,
        default=8,
        help="Number of threads used for M595 FoldX cache precomputation.",
    )
    parser.add_argument(
        "--refresh-foldx-cache",
        action="store_true",
        help="Recompute M595 FoldX cache files even when matching JSON files exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit for the number of M595 entries to evaluate.",
    )
    return parser.parse_args()


def parse_simple_yaml(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip("'\"")
    return data


def get_default_m595_data_root() -> str:
    for candidate in (M595_DATA_ROOT, DDAFFINITY_DATA_ROOT, LEGACY_M595_DATA_ROOT):
        if os.path.isdir(candidate):
            return candidate
    return M595_DATA_ROOT


def load_case_defaults(config_path: Optional[str]) -> Dict[str, str]:
    if not config_path or not os.path.isfile(config_path):
        data_root = get_default_m595_data_root()
        return {
            "pdb_wt_dir": os.path.join(data_root, "wildtype1"),
            "pdb_mt_dir": os.path.join(data_root, "optimized1"),
            "cache_dir": os.path.join(data_root, "entries_cache1"),
        }

    config_dir = os.path.dirname(os.path.abspath(config_path))
    if yaml is not None:
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    else:
        payload = parse_simple_yaml(config_path)

    resolved: Dict[str, str] = {}
    for key in ("pdb_wt_dir", "pdb_mt_dir", "cache_dir"):
        value = payload.get(key)
        if not value:
            continue
        if os.path.isabs(value):
            resolved[key] = value
        else:
            resolved[key] = os.path.abspath(os.path.join(config_dir, value))
    return resolved


def abspath_or_empty(path: Optional[str]) -> str:
    if not path:
        return ""
    return os.path.abspath(path)


def require_path(path: str, label: str) -> str:
    if not path:
        raise ValueError(f"{label} was not resolved.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def validate_runtime_config() -> None:
    feature_dim = getattr(config, "FOLDX_FEATURE_DIM", None)
    if feature_dim != 3:
        raise RuntimeError(
            "m595_blind_prediction_3feature.py must import foldx_3feature_retrain/config.py. "
            f"Expected FOLDX_FEATURE_DIM=3, got {feature_dim!r}. Imported config: {config.__file__}"
        )
    if FOLDX_FEATURE_MODE != "wt_mut_delta":
        raise RuntimeError(
            f"Expected FOLDX_FEATURE_MODE='wt_mut_delta', got {FOLDX_FEATURE_MODE!r}."
        )


def overall_correlations(df: pd.DataFrame) -> Dict[str, float]:
    pearson = df[["ddG", "pred_ddg"]].corr("pearson").iloc[0, 1]
    spearman = df[["ddG", "pred_ddg"]].corr("spearman").iloc[0, 1]
    return {
        "overall_pearson": float(pearson),
        "overall_spearman": float(spearman),
    }


def overall_auroc(df: pd.DataFrame) -> Dict[str, float]:
    labels = (df["ddG"] > 0).to_numpy()
    preds = df["pred_ddg"].to_numpy()
    if len(np.unique(labels)) < 2:
        score = float("nan")
    else:
        score = float(roc_auc_score(labels, preds))
    return {"auroc": score}


def overall_rmse_mae(df: pd.DataFrame) -> Dict[str, float]:
    true = df["ddG"].to_numpy()
    pred = df["pred_ddg"].to_numpy()[:, None]
    reg = LinearRegression().fit(pred, true)
    pred_corrected = reg.predict(pred)
    rmse = float(np.sqrt(((true - pred_corrected) ** 2).mean()))
    mae = float(np.abs(true - pred_corrected).mean())
    return {"rmse": rmse, "mae": mae}


def analyze_all_results(df: pd.DataFrame) -> pd.DataFrame:
    datasets = df["datasets"].unique()
    rows = []
    for dataset in datasets:
        df_this = df[df["datasets"] == dataset]
        row = {"dataset": dataset}
        row.update(overall_correlations(df_this))
        row.update(overall_rmse_mae(df_this))
        row.update(overall_auroc(df_this))
        rows.append(row)
    return pd.DataFrame(rows)


def eval_skempi(df_items: pd.DataFrame, mode: str) -> pd.DataFrame:
    assert mode in ("all", "single", "multiple")
    if mode == "single":
        df_items = df_items.query("num_muts == 1")
    elif mode == "multiple":
        df_items = df_items.query("num_muts > 1")

    df_metrics = analyze_all_results(df_items)
    df_metrics["mode"] = mode
    return df_metrics


def eval_skempi_three_modes(results: pd.DataFrame) -> pd.DataFrame:
    df_all = eval_skempi(results, mode="all")
    df_single = eval_skempi(results, mode="single")
    df_multiple = eval_skempi(results, mode="multiple")
    df_metrics = pd.concat([df_all, df_single, df_multiple], axis=0)
    df_metrics.reset_index(inplace=True, drop=True)
    return df_metrics


def build_model():
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


def load_weights(model, ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint does not contain a state_dict mapping: {ckpt_path}")

    stripped_state = OrderedDict()
    prefixed_state = OrderedDict()
    for key, value in checkpoint.items():
        stripped_state[key.replace("module.", "", 1)] = value
        prefixed_state[key if key.startswith("module.") else f"module.{key}"] = value

    load_errors = []
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        load_errors.append("raw state_dict")

        try:
            model.load_state_dict(stripped_state)
        except RuntimeError:
            load_errors.append("module-prefix stripped state_dict")
            try:
                model.load_state_dict(prefixed_state)
            except RuntimeError as exc:
                load_errors.append("module-prefix added state_dict")
                raise RuntimeError(
                    "Failed to strictly load checkpoint into the 3-feature M595 model. "
                    "Use a checkpoint trained from foldx_3feature_retrain; old 4D checkpoints are incompatible. "
                    f"Tried: {', '.join(load_errors)}. Checkpoint: {ckpt_path}"
                ) from exc

    model.to(device)
    model.eval()
    return model


def build_chain_mapping(group1: List[str], group2: List[str]) -> Dict[str, str]:
    chain_mapping: Dict[str, str] = {}
    if len(group1) == 1:
        chain_mapping["heavy"] = group1[0]
    elif len(group1) >= 2:
        chain_mapping["heavy"] = group1[0]
        chain_mapping["light"] = group1[1]

    if len(group2) >= 1:
        chain_mapping["antigen"] = group2[0]

    return chain_mapping


def extract_sequences_and_mapping(pdb_path: str) -> Tuple[Dict[str, str], Dict[Tuple[str, str], int]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("m595_case", pdb_path)
    model = structure[0]

    seqs: Dict[str, str] = {}
    res_id_to_idx: Dict[Tuple[str, str], int] = {}
    for chain in model:
        residues = [res for res in chain.get_residues() if is_aa(res)]
        if not residues:
            continue
        seq_str = "".join(seq1(res.get_resname()) for res in residues)
        seqs[chain.id] = seq_str
        for idx_in_seq, res in enumerate(residues):
            res_key = (chain.id, str(res.get_id()[1]) + res.get_id()[2].strip())
            res_id_to_idx[res_key] = idx_in_seq

    return seqs, res_id_to_idx


def parse_mutation_token(
    token: str,
    default_chain: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    token = str(token).strip().upper()
    if not token:
        raise ValueError("Empty mutation token.")

    if len(token) >= 4 and token[1].isalpha():
        orig_aa = token[0]
        chain_id = token[1]
        new_aa = token[-1]
        res_num_str = token[2:-1]
    else:
        if default_chain is None:
            raise ValueError(f"Mutation token {token!r} has no chain id and no default chain.")
        orig_aa = token[0]
        chain_id = str(default_chain).strip().upper()
        new_aa = token[-1]
        res_num_str = token[1:-1]

    return orig_aa, chain_id, res_num_str, new_aa


def canonicalize_mutation_string(
    mutation_str: str,
    default_chain: Optional[str] = None,
) -> str:
    normalized_tokens: List[str] = []
    for token in str(mutation_str).split(","):
        orig_aa, chain_id, res_num_str, new_aa = parse_mutation_token(
            token,
            default_chain=default_chain,
        )
        normalized_tokens.append(f"{orig_aa}{chain_id}{res_num_str}{new_aa}")
    return ",".join(normalized_tokens)


def apply_mutations_to_sequences(
    seqs: Dict[str, str],
    res_id_to_idx: Dict[Tuple[str, str], int],
    mutation_str: str,
    default_chain: Optional[str] = None,
) -> Dict[str, str]:
    mutated = dict(seqs)
    for token in str(mutation_str).split(","):
        orig_aa, chain_id, res_num_str, new_aa = parse_mutation_token(
            token,
            default_chain=default_chain,
        )
        if chain_id not in mutated:
            raise KeyError(f"Chain {chain_id} not found when applying mutation {token!r}.")

        res_key = (chain_id, res_num_str)
        if res_key not in res_id_to_idx:
            raise KeyError(f"Residue {chain_id}{res_num_str} not found when applying mutation {token!r}.")

        idx_in_chain = res_id_to_idx[res_key]
        chain_seq = list(mutated[chain_id])
        if idx_in_chain >= len(chain_seq):
            raise IndexError(f"Residue index out of range for mutation {token!r}.")

        observed_wt = chain_seq[idx_in_chain].upper()
        if observed_wt != orig_aa:
            print(
                f"Warning: WT sequence mismatch for mutation {token!r}: "
                f"expected {orig_aa}, observed {observed_wt} at {chain_id}{res_num_str}."
            )

        chain_seq[idx_in_chain] = new_aa
        mutated[chain_id] = "".join(chain_seq)

    return mutated


def build_partner_sequences(
    seqs: Dict[str, str], group1: List[str], group2: List[str]
) -> Tuple[str, str]:
    partner1 = "".join(seqs[c] for c in group1 if c in seqs)
    partner2 = "".join(seqs[c] for c in group2 if c in seqs)
    if not partner1 or not partner2:
        raise ValueError("Failed to build partner sequences from the provided chain groups.")
    return partner1, partner2


def extract_structure_data(
    pdb_path: str,
    group1: List[str],
    group2: List[str],
    mutation_sites: Optional[List[Tuple[str, int]]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    chain_mapping = build_chain_mapping(group1, group2)
    if not chain_mapping:
        return None

    structure_data = cached_structure_extraction(
        pdb_path,
        chain_mapping,
        getattr(config, "USE_ATOM_FEATURES", True),
    )
    if len(structure_data["coords"]) == 0:
        return None

    structure_data = attach_mutation_masks(structure_data, mutation_sites or [])
    structure_data["batch_ids"] = torch.zeros_like(structure_data["segment_ids"])
    return structure_data


def compute_foldx_energy(
    foldx_processor: FoldXProcessor,
    pdb_path: str,
    group1: List[str],
    group2: List[str],
) -> float:
    if not getattr(config, "ENABLE_FOLDX", True):
        return 0.0

    try:
        repaired_pdb_path = foldx_processor.preprocess_pdb(pdb_path)
        return float(
            foldx_processor.extract_features(
                repaired_pdb_path,
                partner1_chains=group1,
                partner2_chains=group2,
            )
        )
    except Exception:
        return 0.0


def infer_one(model, sample: Dict[str, object]) -> float:
    with torch.no_grad():
        pred = model(
            antibody_seqs=sample["antibody_seq"],
            antigen_seqs=sample["antigen_seq"],
            mutant_antibody_seqs=sample["mutant_antibody_seq"],
            mutant_antigen_seqs=sample["mutant_antigen_seq"],
            foldx_energies=sample["foldx_energies"],
            structure_data=sample["structure_data"],
            wt_esm_embedding=sample.get("wt_esm_embedding"),
            mut_esm_embedding=sample.get("mut_esm_embedding"),
            mutation_esm_embedding=sample.get("mutation_esm_embedding"),
            foldx_features=sample.get("foldx_features"),
        )
    return float(pred.squeeze().detach().cpu().item())


def resolve_runtime_paths(args: argparse.Namespace) -> Dict[str, object]:
    defaults = load_case_defaults(args.case_config)
    ckpts: List[str] = []
    for ckpt in args.ckpt:
        ckpt_path = ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config.BASE_DIR, ckpt_path)
        ckpts.append(os.path.abspath(ckpt_path))

    for ckpt in ckpts:
        require_path(ckpt, "checkpoint")

    resolved = {
        "ckpts": ckpts,
        "pdb_wt_dir": abspath_or_empty(args.pdb_wt_dir or defaults.get("pdb_wt_dir")),
        "pdb_mt_dir": abspath_or_empty(args.pdb_mt_dir or defaults.get("pdb_mt_dir")),
        "cache_dir": abspath_or_empty(args.cache_dir or defaults.get("cache_dir")),
        "output_dir": os.path.abspath(args.output_dir),
    }
    require_path(resolved["pdb_wt_dir"], "pdb_wt_dir")
    require_path(resolved["pdb_mt_dir"], "pdb_mt_dir")
    require_path(resolved["cache_dir"], "cache_dir")
    return resolved


def normalize_chain_group(value: Sequence[str]) -> List[str]:
    if isinstance(value, str):
        return [ch for ch in value if ch.strip()]
    return [str(ch).strip() for ch in value if str(ch).strip()]


def load_entries(cache_dir: str) -> List[Dict[str, object]]:
    entries_path = os.path.join(cache_dir, "entries.pkl")
    require_path(entries_path, "entries.pkl")
    with open(entries_path, "rb") as handle:
        entries = pickle.load(handle)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No valid entries were found in {entries_path}")
    return entries


def build_entry_paths(
    entry: Dict[str, object],
    pdb_wt_dir: str,
    pdb_mt_dir: str,
) -> Dict[str, str]:
    pdbcode = str(entry["pdbcode"]).upper()
    wt_path = os.path.join(pdb_wt_dir, f"{pdbcode}.pdb")
    mt_path = os.path.join(pdb_mt_dir, f"{pdbcode}.pdb")
    require_path(wt_path, f"wildtype PDB for {pdbcode}")
    require_path(mt_path, f"mutant PDB for {pdbcode}")
    return {"pdb_wt_path": wt_path, "pdb_mt_path": mt_path}


def m595_foldx_cache_key(
    entry: Dict[str, object],
    paths: Dict[str, str],
    group1: List[str],
    group2: List[str],
) -> str:
    payload = {
        "version": M595_FOLDX_CACHE_VERSION,
        "pdbcode": str(entry["pdbcode"]).upper(),
        "mutation": canonicalize_mutation_string(str(entry["mutstr"])),
        "group1": "".join(group1),
        "group2": "".join(group2),
        "wt_path": os.path.abspath(paths["pdb_wt_path"]),
        "mt_path": os.path.abspath(paths["pdb_mt_path"]),
        "foldx_path": os.path.abspath(config.FOLDX_PATH),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def m595_foldx_cache_path(cache_dir: str, cache_key: str) -> str:
    return os.path.join(cache_dir, f"{cache_key}.json")


def load_m595_foldx_cache(path: str) -> Optional[Dict[str, float]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("version") != M595_FOLDX_CACHE_VERSION:
            return None
        return {
            "wt_foldx_energy": float(payload["wt_foldx_energy"]),
            "mt_foldx_energy": float(payload["mt_foldx_energy"]),
            "foldx_delta_interaction": float(payload["foldx_delta_interaction"]),
            "foldx_buildmodel_ddg": float(payload.get("foldx_buildmodel_ddg", 0.0)),
        }
    except Exception:
        return None


def save_m595_foldx_cache(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def get_thread_foldx_processor(foldx_cache_dir: str) -> FoldXProcessor:
    processor = getattr(_FOLDX_THREAD_LOCAL, "processor", None)
    processor_cache_root = getattr(_FOLDX_THREAD_LOCAL, "processor_cache_root", None)
    if processor is not None and processor_cache_root == foldx_cache_dir:
        return processor

    thread_id = threading.get_ident()
    temp_dir = os.path.join(config.FOLDX_TEMP_DIR, "m595", f"thread_{thread_id}")
    processor_cache_dir = os.path.join(foldx_cache_dir, "_processor_cache", f"thread_{thread_id}")
    processor = FoldXProcessor(
        foldx_path=config.FOLDX_PATH,
        temp_dir=temp_dir,
        cache_dir=processor_cache_dir,
    )
    _FOLDX_THREAD_LOCAL.processor = processor
    _FOLDX_THREAD_LOCAL.processor_cache_root = foldx_cache_dir
    return processor


def compute_one_m595_foldx_cache(
    indexed_entry: Tuple[int, Dict[str, object]],
    pdb_wt_dir: str,
    pdb_mt_dir: str,
    foldx_cache_dir: str,
    refresh: bool,
) -> Dict[str, object]:
    index, entry = indexed_entry
    group1 = normalize_chain_group(entry["group_ligand"])
    group2 = normalize_chain_group(entry["group_receptor"])
    paths = build_entry_paths(entry, pdb_wt_dir, pdb_mt_dir)
    cache_key = m595_foldx_cache_key(entry, paths, group1, group2)
    cache_path = m595_foldx_cache_path(foldx_cache_dir, cache_key)

    if not refresh:
        cached = load_m595_foldx_cache(cache_path)
        if cached is not None:
            return {"index": index, "cache_key": cache_key, "cache_path": cache_path, **cached}

    foldx_processor = get_thread_foldx_processor(foldx_cache_dir)
    wt_foldx_energy = compute_foldx_energy(
        foldx_processor=foldx_processor,
        pdb_path=paths["pdb_wt_path"],
        group1=group1,
        group2=group2,
    )
    mt_foldx_energy = compute_foldx_energy(
        foldx_processor=foldx_processor,
        pdb_path=paths["pdb_mt_path"],
        group1=group1,
        group2=group2,
    )
    foldx_delta_interaction = mt_foldx_energy - wt_foldx_energy
    foldx_buildmodel_ddg = 0.0
    payload: Dict[str, object] = {
        "version": M595_FOLDX_CACHE_VERSION,
        "index": index,
        "pdbcode": str(entry["pdbcode"]).upper(),
        "mutation": canonicalize_mutation_string(str(entry["mutstr"])),
        "group1": "".join(group1),
        "group2": "".join(group2),
        "pdb_wt_path": os.path.abspath(paths["pdb_wt_path"]),
        "pdb_mt_path": os.path.abspath(paths["pdb_mt_path"]),
        "wt_foldx_energy": wt_foldx_energy,
        "mt_foldx_energy": mt_foldx_energy,
        "foldx_delta_interaction": foldx_delta_interaction,
        "foldx_buildmodel_ddg": foldx_buildmodel_ddg,
    }
    save_m595_foldx_cache(cache_path, payload)
    return {"index": index, "cache_key": cache_key, "cache_path": cache_path, **payload}


def precompute_m595_foldx_cache(
    entries: List[Dict[str, object]],
    pdb_wt_dir: str,
    pdb_mt_dir: str,
    foldx_cache_dir: str,
    workers: int,
    refresh: bool,
) -> Dict[int, Dict[str, object]]:
    os.makedirs(foldx_cache_dir, exist_ok=True)
    workers = max(1, int(workers))
    results: Dict[int, Dict[str, object]] = {}

    if workers == 1:
        iterator = (
            compute_one_m595_foldx_cache(
                indexed_entry,
                pdb_wt_dir=pdb_wt_dir,
                pdb_mt_dir=pdb_mt_dir,
                foldx_cache_dir=foldx_cache_dir,
                refresh=refresh,
            )
            for indexed_entry in enumerate(entries)
        )
        for result in tqdm(iterator, total=len(entries), desc="Caching M595 FoldX", dynamic_ncols=True):
            results[int(result["index"])] = result
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                compute_one_m595_foldx_cache,
                indexed_entry,
                pdb_wt_dir,
                pdb_mt_dir,
                foldx_cache_dir,
                refresh,
            )
            for indexed_entry in enumerate(entries)
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Caching M595 FoldX ({workers} workers)",
            dynamic_ncols=True,
        ):
            result = future.result()
            results[int(result["index"])] = result

    return results


def prepare_cached_inputs(
    entries: List[Dict[str, object]],
    pdb_wt_dir: str,
    pdb_mt_dir: str,
    structure_source: str,
    device: str,
    foldx_cache: Dict[int, Dict[str, object]],
    esm_helper: Optional[ESMEmbeddingHelper],
) -> List[Dict[str, object]]:
    prepared: List[Dict[str, object]] = []
    for entry_index, entry in enumerate(tqdm(entries, desc="Preparing M595 inputs", dynamic_ncols=True)):
        group1 = normalize_chain_group(entry["group_ligand"])
        group2 = normalize_chain_group(entry["group_receptor"])
        paths = build_entry_paths(entry, pdb_wt_dir, pdb_mt_dir)

        wt_seqs, wt_res_id_to_idx = extract_sequences_and_mapping(paths["pdb_wt_path"])
        normalized_mutation = canonicalize_mutation_string(str(entry["mutstr"]))
        mutant_seqs = apply_mutations_to_sequences(
            wt_seqs,
            wt_res_id_to_idx,
            normalized_mutation,
        )
        mutation_sites = []
        for token in normalized_mutation.split(","):
            _, chain_id, res_num_str, _ = parse_mutation_token(token)
            res_key = (chain_id, res_num_str)
            if res_key in wt_res_id_to_idx:
                mutation_sites.append((chain_id, wt_res_id_to_idx[res_key]))

        wt_partner1, wt_partner2 = build_partner_sequences(wt_seqs, group1, group2)
        mt_partner1, mt_partner2 = build_partner_sequences(mutant_seqs, group1, group2)
        wt_esm_embedding = None
        mut_esm_embedding = None
        mutation_esm_embedding = None
        if esm_helper is not None:
            wt_esm_embedding = esm_helper.encode_complex(wt_partner1, wt_partner2).unsqueeze(0).cpu()
            mut_esm_embedding = esm_helper.encode_complex(mt_partner1, mt_partner2).unsqueeze(0).cpu()
            mutation_esm_embedding = esm_helper.encode_mutation_context(
                wt_partner1,
                wt_partner2,
                mt_partner1,
                mt_partner2,
            ).unsqueeze(0).cpu()

        structure_pdb_path = paths["pdb_mt_path"] if structure_source == "mt" else paths["pdb_wt_path"]
        structure_data = extract_structure_data(
            structure_pdb_path,
            group1,
            group2,
            mutation_sites=mutation_sites,
        )

        cached_foldx = foldx_cache.get(entry_index)
        if cached_foldx is None:
            raise RuntimeError(f"Missing M595 FoldX cache for entry index {entry_index}.")
        wt_foldx_energy = float(cached_foldx["wt_foldx_energy"])
        mt_foldx_energy = float(cached_foldx["mt_foldx_energy"])
        foldx_delta_interaction = float(cached_foldx["foldx_delta_interaction"])
        foldx_buildmodel_ddg = float(cached_foldx.get("foldx_buildmodel_ddg", 0.0))
        foldx_features = torch.tensor(
            [
                [
                    wt_foldx_energy,
                    mt_foldx_energy,
                    foldx_delta_interaction,
                ]
            ],
            dtype=torch.float32,
        )
        expected_dim = getattr(config, "FOLDX_FEATURE_DIM", 3)
        if foldx_features.shape[-1] != expected_dim:
            raise RuntimeError(
                f"M595 3-feature script built {foldx_features.shape[-1]} FoldX features, "
                f"but config.FOLDX_FEATURE_DIM={expected_dim}."
            )

        sample: Dict[str, object] = {
            "complex": entry.get("complex", ""),
            "pdbcode": entry["pdbcode"],
            "mutation": normalized_mutation,
            "num_muts": entry["num_muts"],
            "experimental_ddg": float(entry["ddG"]) if entry.get("ddG") is not None else None,
            "pdb_wt_path": paths["pdb_wt_path"],
            "pdb_mt_path": paths["pdb_mt_path"],
            "chain_group_1": "".join(group1),
            "chain_group_2": "".join(group2),
            "structure_source": structure_source,
            "sequence_build_source": "wt_plus_mutation_string",
            "antibody_seq": [wt_partner1],
            "antigen_seq": [wt_partner2],
            "mutant_antibody_seq": [mt_partner1],
            "mutant_antigen_seq": [mt_partner2],
            "foldx_energy": wt_foldx_energy,
            "foldx_mut_energy": mt_foldx_energy,
            "foldx_delta_interaction": foldx_delta_interaction,
            "foldx_buildmodel_ddg": foldx_buildmodel_ddg,
            "foldx_feature_mode": FOLDX_FEATURE_MODE,
            "foldx_features": foldx_features,
            "foldx_energies": torch.tensor([wt_foldx_energy], dtype=torch.float32),
            "structure_data": structure_data,
            "wt_esm_embedding": wt_esm_embedding,
            "mut_esm_embedding": mut_esm_embedding,
            "mutation_esm_embedding": mutation_esm_embedding,
        }
        prepared.append(sample)

    return prepared


def move_structure_data_to_device(
    structure_data: Optional[Dict[str, torch.Tensor]],
    device: str,
) -> Optional[Dict[str, torch.Tensor]]:
    if structure_data is None:
        return None
    moved: Dict[str, torch.Tensor] = {}
    for key, value in structure_data.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def move_sample_to_device(sample: Dict[str, object], device: str) -> Dict[str, object]:
    moved = dict(sample)
    moved["foldx_energies"] = sample["foldx_energies"].to(device)  # type: ignore[index]
    if sample.get("foldx_features") is not None:
        moved["foldx_features"] = sample["foldx_features"].to(device)  # type: ignore[index]
    moved["structure_data"] = move_structure_data_to_device(sample["structure_data"], device)  # type: ignore[index]
    if sample["wt_esm_embedding"] is not None:
        moved["wt_esm_embedding"] = sample["wt_esm_embedding"].to(device)  # type: ignore[index]
    if sample["mut_esm_embedding"] is not None:
        moved["mut_esm_embedding"] = sample["mut_esm_embedding"].to(device)  # type: ignore[index]
    if sample.get("mutation_esm_embedding") is not None:
        moved["mutation_esm_embedding"] = sample["mutation_esm_embedding"].to(device)  # type: ignore[index]
    return moved


def load_local_model(ckpt_path: str, device: str):
    model = build_model()
    model = load_weights(model, ckpt_path, device)
    model.eval()
    return model


def run_checkpoint_predictions(
    ckpt_path: str,
    prepared_samples: List[Dict[str, object]],
    device: str,
) -> List[Dict[str, object]]:
    model = load_local_model(ckpt_path, device)
    ckpt_name = os.path.basename(ckpt_path)
    rows: List[Dict[str, object]] = []

    for sample in tqdm(prepared_samples, desc=f"Predicting {ckpt_name}", dynamic_ncols=True):
        model_sample = move_sample_to_device(sample, device)
        pred_ddg = infer_one(model, model_sample)
        rows.append(
            {
                "checkpoint": ckpt_name,
                "complex": sample["complex"],
                "pdbcode": sample["pdbcode"],
                "mutstr": sample["mutation"],
                "num_muts": sample["num_muts"],
                "ddG": sample["experimental_ddg"],
                "pred_ddg": pred_ddg,
                "foldx_energy": sample["foldx_energy"],
                "foldx_mut_energy": sample["foldx_mut_energy"],
                "foldx_delta_interaction": sample["foldx_delta_interaction"],
                "foldx_buildmodel_ddg": sample["foldx_buildmodel_ddg"],
                "foldx_feature_mode": sample["foldx_feature_mode"],
                "structure_source": sample["structure_source"],
                "sequence_build_source": sample.get("sequence_build_source", ""),
                "pdb_wt_path": sample["pdb_wt_path"],
                "pdb_mt_path": sample["pdb_mt_path"],
                "chain_group_1": sample["chain_group_1"],
                "chain_group_2": sample["chain_group_2"],
            }
        )

    return rows


def resolve_aggregate_mode(mode: str, num_ckpts: int) -> str:
    if mode == "auto":
        return "identity" if num_ckpts == 1 else "ddaffinity"
    if mode == "ddaffinity" and num_ckpts == 1:
        print("Warning: only one checkpoint was provided, so DDAffinity aggregation falls back to identity.")
        return "identity"
    return mode


def aggregate_predictions(raw_results: pd.DataFrame, aggregate_mode: str) -> pd.DataFrame:
    grouped = raw_results.groupby("mutstr", as_index=False).agg(
        ddG=("ddG", "mean"),
        num_muts=("num_muts", "mean"),
        pred_ddg_mean=("pred_ddg", "mean"),
        pred_ddg_max=("pred_ddg", "max"),
        pred_ddg_min=("pred_ddg", "min"),
        num_predictions=("pred_ddg", "size"),
        complex=("complex", "first"),
        pdbcode=("pdbcode", "first"),
        foldx_energy=("foldx_energy", "mean"),
        foldx_mut_energy=("foldx_mut_energy", "mean"),
        foldx_delta_interaction=("foldx_delta_interaction", "mean"),
        foldx_buildmodel_ddg=("foldx_buildmodel_ddg", "mean"),
        foldx_feature_mode=("foldx_feature_mode", "first"),
        chain_group_1=("chain_group_1", "first"),
        chain_group_2=("chain_group_2", "first"),
        structure_source=("structure_source", "first"),
        sequence_build_source=("sequence_build_source", "first"),
    )

    if aggregate_mode == "identity":
        grouped["pred_ddg"] = grouped["pred_ddg_mean"]
    elif aggregate_mode == "mean":
        grouped["pred_ddg"] = grouped["pred_ddg_mean"]
    elif aggregate_mode == "ddaffinity":
        grouped["pred_ddg"] = -(grouped["pred_ddg_max"] - grouped["pred_ddg_min"]) / 2.0
    else:
        raise ValueError(f"Unsupported aggregate mode: {aggregate_mode}")

    grouped = grouped.sort_values(["pred_ddg", "mutstr"], ascending=[True, True]).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1
    grouped["rank_percentile"] = grouped["rank"] / len(grouped)
    grouped["datasets"] = "case_study"
    return grouped


def maybe_write_metrics(
    aggregated_results: pd.DataFrame,
    metrics_path: str,
    skip_metrics: bool,
) -> Optional[pd.DataFrame]:
    if skip_metrics:
        return None
    valid = aggregated_results["ddG"].notna()
    if not valid.any():
        return None
    metrics = eval_skempi_three_modes(aggregated_results.loc[valid].copy())
    metrics.to_csv(metrics_path, index=False)
    return metrics


def print_topk(aggregated_results: pd.DataFrame, topk: int) -> None:
    preview = aggregated_results.head(topk)[
        ["mutstr", "pred_ddg", "ddG", "num_muts", "rank_percentile"]
    ]
    print("")
    print(f"Top {min(topk, len(aggregated_results))} M595 candidates:")
    print(preview.to_string(index=False))


def main() -> None:
    validate_runtime_config()
    args = parse_args()
    paths = resolve_runtime_paths(args)
    os.makedirs(paths["output_dir"], exist_ok=True)  # type: ignore[index]

    print(f"Using checkpoints : {paths['ckpts']}")
    print(f"Using wildtype    : {paths['pdb_wt_dir']}")
    print(f"Using mutant      : {paths['pdb_mt_dir']}")
    print(f"Using cache       : {paths['cache_dir']}")
    print(f"FoldX cache dir   : {os.path.abspath(args.foldx_cache_dir)}")
    print(f"FoldX workers     : {args.foldx_workers}")
    print(f"FoldX feature mode: {FOLDX_FEATURE_MODE} ([wt_energy, mt_energy, mt-wt])")
    print(f"Structure source  : {args.structure_source}")
    print(f"ESM precomputed   : {getattr(config, 'USE_PRECOMPUTED_ESM', False)}")
    print("Sequence path     : WT sequence + mutation string (not mutant PDB sequence)")

    entries = load_entries(paths["cache_dir"])  # type: ignore[arg-type]
    if args.limit is not None:
        entries = entries[: max(0, args.limit)]
        print(f"Debug limit       : {len(entries)} entries")

    foldx_cache = precompute_m595_foldx_cache(
        entries=entries,
        pdb_wt_dir=paths["pdb_wt_dir"],  # type: ignore[arg-type]
        pdb_mt_dir=paths["pdb_mt_dir"],  # type: ignore[arg-type]
        foldx_cache_dir=os.path.abspath(args.foldx_cache_dir),
        workers=args.foldx_workers,
        refresh=args.refresh_foldx_cache,
    )
    esm_helper = None
    if getattr(config, "USE_PRECOMPUTED_ESM", False):
        esm_helper = ESMEmbeddingHelper(config.ESM_MODEL_NAME, torch.device(args.device))

    prepared_samples = prepare_cached_inputs(
        entries=entries,
        pdb_wt_dir=paths["pdb_wt_dir"],  # type: ignore[arg-type]
        pdb_mt_dir=paths["pdb_mt_dir"],  # type: ignore[arg-type]
        structure_source=args.structure_source,
        device=args.device,
        foldx_cache=foldx_cache,
        esm_helper=esm_helper,
    )

    metadata_rows = [
        {
            "complex": sample["complex"],
            "pdbcode": sample["pdbcode"],
            "mutation": sample["mutation"],
            "num_muts": sample["num_muts"],
            "experimental_ddg": sample["experimental_ddg"],
            "pdb_wt_path": sample["pdb_wt_path"],
            "pdb_mt_path": sample["pdb_mt_path"],
            "chain_group_1": sample["chain_group_1"],
            "chain_group_2": sample["chain_group_2"],
            "structure_source": sample["structure_source"],
            "sequence_build_source": sample.get("sequence_build_source", ""),
            "foldx_energy": sample["foldx_energy"],
            "foldx_mut_energy": sample["foldx_mut_energy"],
            "foldx_delta_interaction": sample["foldx_delta_interaction"],
            "foldx_buildmodel_ddg": sample["foldx_buildmodel_ddg"],
            "foldx_feature_mode": sample["foldx_feature_mode"],
        }
        for sample in prepared_samples
    ]

    metadata_path = os.path.join(paths["output_dir"], "m595_metadata.csv")  # type: ignore[index]
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    raw_rows: List[Dict[str, object]] = []
    for ckpt_path in paths["ckpts"]:  # type: ignore[index]
        raw_rows.extend(
            run_checkpoint_predictions(
                ckpt_path=ckpt_path,
                prepared_samples=prepared_samples,
                device=args.device,
            )
        )

    raw_results = pd.DataFrame(raw_rows)
    aggregate_mode = resolve_aggregate_mode(args.aggregate_mode, len(paths["ckpts"]))  # type: ignore[index]
    aggregated_results = aggregate_predictions(raw_results, aggregate_mode)

    raw_path = os.path.join(paths["output_dir"], "m595_raw_predictions_3feature.csv")  # type: ignore[index]
    aggregated_path = os.path.join(paths["output_dir"], "m595_aggregated_predictions_3feature.csv")  # type: ignore[index]
    metrics_path = os.path.join(paths["output_dir"], "m595_three_mode_metrics_3feature.csv")  # type: ignore[index]

    raw_results.to_csv(raw_path, index=False)
    aggregated_results.to_csv(aggregated_path, index=False)

    print(f"Metadata saved to              : {metadata_path}")
    print(f"Raw predictions saved to       : {raw_path}")
    print(f"Aggregated predictions saved to: {aggregated_path}")
    print(f"Aggregation mode               : {aggregate_mode}")

    metrics = maybe_write_metrics(
        aggregated_results=aggregated_results,
        metrics_path=metrics_path,
        skip_metrics=args.skip_metrics,
    )
    if metrics is not None:
        print(f"Three-mode metrics saved to    : {metrics_path}")
        print("")
        print(metrics.to_string(index=False))
    else:
        print("Three-mode metrics were skipped or ddG labels were unavailable.")

    print_topk(aggregated_results, args.topk)


if __name__ == "__main__":
    main()
