import argparse
import csv
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

import config
from case_study_foldx import CaseStudyFoldXBuilder
from data_loader import (
    attach_mutation_masks,
    attach_structure_chain_indices,
    build_all_chain_mapping,
    build_structure_residue_indices,
    cached_structure_extraction,
    ensure_structure_residue_ids,
)
from esm_local_tokens import (
    build_local_esm_context,
    pack_preselected_esm_tokens,
    pool_packed_mutation_esm_features,
)
from eval_pth_metrics import build_model, load_weights
from foldx_processor import FoldXProcessor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


CASE_FOLDX_FEATURE_CACHE_VERSION = 3
MAX_ESM_RESIDUES = 1022


class ESMEmbeddingHelper:
    def __init__(self, model_name: str, device: torch.device):
        from esm.pretrained import load_model_and_alphabet

        self.model, self.alphabet = load_model_and_alphabet(model_name)
        self.model.eval().to(device)
        self.device = device

    def encode_complex(self, partner1_seq: str, partner2_seq: str) -> torch.Tensor:
        pooled, _, _ = self.encode_complex_tokens(partner1_seq, partner2_seq)
        return pooled

    def encode_complex_tokens(self, partner1_seq: str, partner2_seq: str):
        batch_converter = self.alphabet.get_batch_converter()
        sequence = (partner1_seq + partner2_seq)[:MAX_ESM_RESIDUES]
        data = [("complex", sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.model.num_layers],
                return_contacts=False,
            )

        token_representations = results["representations"][self.model.num_layers]
        seq_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        token_representations = token_representations[:, 1:-1]
        actual_len = int(seq_lens[0].item()) - 2
        pooled = token_representations[0, :actual_len].mean(0)
        return pooled.detach(), token_representations[0, :actual_len].detach(), actual_len

    def encode_mutation_context(self, wt_partner1: str, wt_partner2: str, mut_partner1: str, mut_partner2: str) -> torch.Tensor:
        features = self.encode_local_token_pair(
            wt_partner1,
            wt_partner2,
            mut_partner1,
            mut_partner2,
        )
        return pool_packed_mutation_esm_features(features).detach()

    def encode_local_token_pair(
        self,
        wt_partner1: str,
        wt_partner2: str,
        mut_partner1: str,
        mut_partner2: str,
        expected_mutation_count: Optional[int] = None,
        mutation_positions: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        wt_sequence = wt_partner1 + wt_partner2
        mutant_sequence = mut_partner1 + mut_partner2
        wt_pooled, wt_tokens, _ = self.encode_complex_tokens(
            wt_partner1,
            wt_partner2,
        )
        mutant_pooled, mutant_tokens, _ = self.encode_complex_tokens(
            mut_partner1,
            mut_partner2,
        )
        context = build_local_esm_context(
            wt_sequence=wt_sequence,
            mutant_sequence=mutant_sequence,
            radius=getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8),
            max_tokens=getattr(config, "ESM_LOCAL_MAX_TOKENS", 32),
            max_context_length=MAX_ESM_RESIDUES,
            expected_mutation_count=expected_mutation_count,
            mutation_positions=mutation_positions,
        )
        if context["context_start"] != 0:
            _, wt_tokens, _ = self.encode_complex_tokens(
                context["wt_context_sequence"],
                "",
            )
            _, mutant_tokens, _ = self.encode_complex_tokens(
                context["mutant_context_sequence"],
                "",
            )
        packed = pack_preselected_esm_tokens(
            wt_context_tokens=wt_tokens,
            mutant_context_tokens=mutant_tokens,
            context_token_indices=context["context_token_indices"],
            selected_positions=context["selected_positions"],
            mutation_positions=context["mutation_positions"],
            max_tokens=getattr(config, "ESM_LOCAL_MAX_TOKENS", 32),
        )
        return {
            "wt_esm_embedding": wt_pooled.detach(),
            "mut_esm_embedding": mutant_pooled.detach(),
            **{key: value.detach() for key, value in packed.items()},
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run case-study inference with best_model.pth-compatible inputs.")
    parser.add_argument("--case-csv", required=True, help="Path to metadata CSV.")
    parser.add_argument("--ckpt", default=config.BEST_MODEL_PATH, help="Checkpoint path. Default: best_model.pth.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path for predictions.")
    parser.add_argument("--task", choices=["rbd_ddg", "antibody_opt"], required=True)
    parser.add_argument(
        "--skip-wt-aa-check",
        action="store_true",
        help="Relax WT amino-acid checks when attaching mutation-aware structure features.",
    )
    return parser.parse_args()


def read_metadata(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_chain_group(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []
    if "," in value:
        return [x.strip() for x in value.split(",") if x.strip()]
    return list(value)


def parse_mutation_token(token: str, default_chain: Optional[str] = None) -> Tuple[str, str, str, str]:
    token = token.strip()
    if not token:
        raise ValueError("Empty mutation token")

    # Supports A123B or XA123B where X is chain id.
    if len(token) >= 4 and token[1].isalpha():
        orig_aa = token[0]
        chain_id = token[1]
        new_aa = token[-1]
        res_num_str = token[2:-1]
    else:
        if default_chain is None:
            raise ValueError(f"Mutation {token} has no chain id and no default chain is provided")
        orig_aa = token[0]
        chain_id = default_chain
        new_aa = token[-1]
        res_num_str = token[1:-1]

    return orig_aa, chain_id, res_num_str, new_aa


def build_chain_mapping(group1: List[str], group2: List[str]) -> Dict[str, str]:
    return build_all_chain_mapping(group1, group2)


def extract_sequences_and_mapping(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("case", pdb_path)
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


def apply_mutations_to_sequences(
    seqs: Dict[str, str],
    res_id_to_idx: Dict[Tuple[str, str], int],
    mutation_str: str,
    default_chain: Optional[str] = None,
) -> Dict[str, str]:
    mutated = dict(seqs)
    for token in mutation_str.split(","):
        orig_aa, chain_id, res_num_str, new_aa = parse_mutation_token(token, default_chain=default_chain)
        if chain_id not in mutated:
            raise KeyError(f"Chain {chain_id} not found in structure")
        res_key = (chain_id, res_num_str)
        if res_key not in res_id_to_idx:
            raise KeyError(f"Residue {chain_id}{res_num_str} not found in structure")
        idx_in_chain = res_id_to_idx[res_key]
        chain_seq = list(mutated[chain_id])
        if idx_in_chain >= len(chain_seq):
            raise IndexError(f"Residue index out of range for mutation {token}")
        chain_seq[idx_in_chain] = new_aa
        mutated[chain_id] = "".join(chain_seq)
    return mutated


def build_partner_sequences(seqs: Dict[str, str], group1: List[str], group2: List[str]) -> Tuple[str, str]:
    partner1 = "".join(seqs[c] for c in group1 if c in seqs)
    partner2 = "".join(seqs[c] for c in group2 if c in seqs)
    if not partner1 or not partner2:
        raise ValueError("Failed to build partner sequences from the provided chain groups")
    return partner1, partner2


def build_partner_mutation_positions(
    seqs: Dict[str, str],
    res_id_to_idx: Dict[Tuple[str, str], int],
    group1: List[str],
    group2: List[str],
    mutation_sites: List[Tuple[str, str]],
) -> List[int]:
    """Map annotated PDB mutation sites onto the concatenated partner sequence."""
    offsets: Dict[str, int] = {}
    cursor = 0
    for chain_id in group1:
        offsets[chain_id] = cursor
        cursor += len(seqs.get(chain_id, ""))
    for chain_id in group2:
        offsets[chain_id] = cursor
        cursor += len(seqs.get(chain_id, ""))

    positions: List[int] = []
    for chain_id, residue_id in mutation_sites:
        if chain_id not in offsets:
            raise KeyError(f"Mutation chain {chain_id} is not in the partner groups.")
        key = (chain_id, residue_id)
        if key not in res_id_to_idx:
            raise KeyError(f"Mutation residue {chain_id}{residue_id} was not indexed.")
        positions.append(offsets[chain_id] + int(res_id_to_idx[key]))
    return positions


def canonicalize_mutation_string(
    mutation_str: str,
    default_chain: Optional[str] = None,
) -> str:
    normalized_tokens = []
    for token in mutation_str.split(","):
        orig_aa, chain_id, res_num_str, new_aa = parse_mutation_token(
            token,
            default_chain=default_chain,
        )
        normalized_tokens.append(f"{orig_aa}{chain_id}{res_num_str}{new_aa}")
    return ",".join(normalized_tokens)


def extract_structure_data(
    pdb_path: str,
    group1: List[str],
    group2: List[str],
    mutation_sites: Optional[List[Tuple[str, str]]] = None,
    mutation_types: Optional[Dict[Tuple[str, str], Tuple[str, str]]] = None,
    strict_wt_check: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    chain_mapping = build_chain_mapping(group1, group2)
    if not chain_mapping:
        return None

    structure_data = cached_structure_extraction(
        pdb_path,
        chain_mapping,
        getattr(config, "USE_ATOM_FEATURES", True),
    )
    structure_data = ensure_structure_residue_ids(structure_data, pdb_path, chain_mapping)
    if len(structure_data["coords"]) == 0:
        return None

    structure_data = attach_mutation_masks(
        structure_data,
        mutation_sites or [],
        mutation_types=mutation_types,
        strict_wt_check=strict_wt_check,
    )
    structure_data, _ = attach_structure_chain_indices(structure_data)
    structure_data["batch_ids"] = torch.zeros_like(structure_data["segment_ids"])
    ca_mask, residue_uid, _ = build_structure_residue_indices(structure_data)
    structure_data["ca_mask"] = ca_mask
    structure_data["residue_uid"] = residue_uid
    return structure_data


def compute_foldx_energy(
    foldx_processor: FoldXProcessor,
    pdb_path: str,
    group1: List[str],
    group2: List[str],
) -> float:
    if not getattr(config, "ENABLE_FOLDX", True):
        raise RuntimeError("ENABLE_FOLDX must be true for 3-feature case-study inference.")

    repaired_pdb_path = foldx_processor.preprocess_pdb(pdb_path)
    return float(
        foldx_processor.extract_features(
            repaired_pdb_path,
            partner1_chains=group1,
            partner2_chains=group2,
        )
    )


def case_foldx_feature_cache_key(
    pdb_path: str,
    chain_group_1: str,
    chain_group_2: str,
    normalized_mutation: str,
) -> str:
    foldx_path = os.path.abspath(config.FOLDX_PATH)
    foldx_size = os.path.getsize(foldx_path) if os.path.exists(foldx_path) else "missing"
    text = "|".join(
        [
            f"{os.path.basename(foldx_path)}:{foldx_size}",
            os.path.abspath(pdb_path),
            chain_group_1,
            chain_group_2,
            normalized_mutation,
            getattr(config, "FOLDX_VERSION", "foldx"),
        ]
    )
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_case_foldx_feature_cache(
    pdb_path: str,
    chain_group_1: str,
    chain_group_2: str,
    normalized_mutation: str,
) -> Optional[Dict[str, object]]:
    cache_key = case_foldx_feature_cache_key(
        pdb_path=pdb_path,
        chain_group_1=chain_group_1,
        chain_group_2=chain_group_2,
        normalized_mutation=normalized_mutation,
    )
    cache_path = os.path.join(
        config.FOLDX_CACHE_DIR,
        "case_study_mutants",
        f"{cache_key}_FoldXFeatures.json",
    )
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") != "ok":
            return None
        if int(payload.get("case_foldx_feature_cache_version", 0)) < CASE_FOLDX_FEATURE_CACHE_VERSION:
            return None
        return payload
    except Exception:
        return None


def save_case_foldx_feature_cache(payload: Dict[str, object]) -> None:
    cache_key = case_foldx_feature_cache_key(
        pdb_path=str(payload["pdb_path"]),
        chain_group_1=str(payload["chain_group_1"]),
        chain_group_2=str(payload["chain_group_2"]),
        normalized_mutation=str(payload["normalized_mutation"]),
    )
    cache_path = os.path.join(
        config.FOLDX_CACHE_DIR,
        "case_study_mutants",
        f"{cache_key}_FoldXFeatures.json",
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, cache_path)


def prepare_model_inputs(
    row: Dict[str, str],
    foldx_processor: FoldXProcessor,
    foldx_builder: CaseStudyFoldXBuilder,
    esm_helper: Optional[ESMEmbeddingHelper],
    device: torch.device,
    strict_wt_check: bool = True,
) -> Dict[str, object]:
    pdb_path = os.path.abspath(row["pdb_path"])
    group1 = normalize_chain_group(row["chain_group_1"])
    group2 = normalize_chain_group(row["chain_group_2"])
    default_chain = row.get("mutation_chain") or (group1[0] if len(group1) == 1 else None)
    normalized_mutation = canonicalize_mutation_string(
        row["mutation"],
        default_chain=default_chain,
    )

    seqs, res_id_to_idx = extract_sequences_and_mapping(pdb_path)
    wt_partner1, wt_partner2 = build_partner_sequences(seqs, group1, group2)
    mutation_sites = []
    mutation_types = {}
    for token in normalized_mutation.split(","):
        orig_aa, chain_id, res_num_str, new_aa = parse_mutation_token(
            token,
            default_chain=default_chain,
        )
        res_key = (chain_id, res_num_str)
        if res_key in res_id_to_idx:
            mutation_sites.append((chain_id, res_num_str))
            mutation_types[(chain_id, res_num_str)] = (orig_aa, new_aa)
    is_noop_mutation = bool(mutation_types) and all(
        orig_aa.upper() == new_aa.upper()
        for orig_aa, new_aa in mutation_types.values()
    )

    mutated_seqs = apply_mutations_to_sequences(
        seqs,
        res_id_to_idx,
        normalized_mutation,
        default_chain=default_chain,
    )
    mut_partner1, mut_partner2 = build_partner_sequences(mutated_seqs, group1, group2)

    structure_pdb_path = pdb_path
    mutant_pdb_path = ""
    foldx_buildmodel_ddg = 0.0
    cached_foldx_features = None
    if not is_noop_mutation:
        cached_foldx_features = load_case_foldx_feature_cache(
            pdb_path=pdb_path,
            chain_group_1=str(row["chain_group_1"]),
            chain_group_2=str(row["chain_group_2"]),
            normalized_mutation=normalized_mutation,
    )
    should_build_mutant_pdb = (
        not is_noop_mutation
        and getattr(config, "ENABLE_FOLDX", True)
        and (
            getattr(config, "CASE_STUDY_BUILD_MUTANT_PDB", False)
            or getattr(config, "USE_MUTATION_FOLDX_FEATURES", False)
        )
        and cached_foldx_features is None
    )
    if cached_foldx_features is not None:
        mutant_pdb_path = str(cached_foldx_features.get("mutant_pdb_path", ""))
        foldx_buildmodel_ddg = float(cached_foldx_features.get("foldx_buildmodel_ddg") or 0.0)
    if should_build_mutant_pdb:
        repaired_wt_pdb_path = foldx_processor.preprocess_pdb(pdb_path)
        mutant_pdb_path, buildmodel_ddg = foldx_builder.build_mutant_model_with_ddg(
            repaired_pdb_path=repaired_wt_pdb_path,
            foldx_mutation_spec=normalized_mutation,
        )
        foldx_buildmodel_ddg = float(buildmodel_ddg or 0.0)

    structure_data = extract_structure_data(
        structure_pdb_path,
        group1,
        group2,
        mutation_sites,
        mutation_types,
        strict_wt_check=strict_wt_check,
    )
    if structure_data is not None:
        for key, value in structure_data.items():
            if isinstance(value, torch.Tensor):
                structure_data[key] = value.to(device)

    if is_noop_mutation:
        foldx_energy = compute_foldx_energy(
            foldx_processor,
            structure_pdb_path,
            group1,
            group2,
        )
        mut_foldx_energy = foldx_energy
        mutant_pdb_path = structure_pdb_path
        foldx_buildmodel_ddg = 0.0
    elif cached_foldx_features is not None:
        foldx_energy = float(cached_foldx_features.get("foldx_energy", 0.0))
        mut_foldx_energy = float(cached_foldx_features.get("foldx_mut_energy", foldx_energy))
    else:
        foldx_energy = compute_foldx_energy(foldx_processor, structure_pdb_path, group1, group2)
        mut_foldx_energy = foldx_energy
        if mutant_pdb_path:
            mut_foldx_energy = compute_foldx_energy(foldx_processor, mutant_pdb_path, group1, group2)
        else:
            raise RuntimeError(
                "3-feature case-study inference requires a mutant PDB or cached mutant FoldX energy. "
                f"Set CASE_STUDY_BUILD_MUTANT_PDB=1 or precompute cache for {normalized_mutation}."
            )
    foldx_tensor = torch.tensor([foldx_energy], dtype=torch.float32, device=device)
    foldx_features = torch.tensor(
        [[foldx_energy, mut_foldx_energy, mut_foldx_energy - foldx_energy]],
        dtype=torch.float32,
        device=device,
    )
    if (
        not is_noop_mutation
        and cached_foldx_features is None
        and mutant_pdb_path
    ):
        save_case_foldx_feature_cache(
            {
                "status": "ok",
                "case_foldx_feature_cache_version": CASE_FOLDX_FEATURE_CACHE_VERSION,
                "pdb_path": pdb_path,
                "chain_group_1": str(row["chain_group_1"]),
                "chain_group_2": str(row["chain_group_2"]),
                "normalized_mutation": normalized_mutation,
                "mutation": row["mutation"],
                "foldx_energy": foldx_energy,
                "foldx_mut_energy": mut_foldx_energy,
                "foldx_delta_interaction": mut_foldx_energy - foldx_energy,
                "foldx_buildmodel_ddg": foldx_buildmodel_ddg,
                "foldx_feature_mode": getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta"),
                "mutant_pdb_path": mutant_pdb_path,
                "foldx_version": getattr(config, "FOLDX_VERSION", "foldx"),
                "foldx_path": os.path.abspath(config.FOLDX_PATH),
            }
        )

    wt_esm_embedding = None
    mut_esm_embedding = None
    wt_esm_window_tokens = None
    mut_esm_window_tokens = None
    esm_window_padding_mask = None
    esm_window_mutation_mask = None
    if esm_helper is not None:
        esm_mutation_positions = build_partner_mutation_positions(
            seqs,
            res_id_to_idx,
            group1,
            group2,
            mutation_sites,
        )
        esm_features = esm_helper.encode_local_token_pair(
            wt_partner1,
            wt_partner2,
            mut_partner1,
            mut_partner2,
            expected_mutation_count=len(mutation_sites),
            mutation_positions=esm_mutation_positions,
        )
        wt_esm_embedding = esm_features["wt_esm_embedding"].unsqueeze(0).to(device)
        mut_esm_embedding = esm_features["mut_esm_embedding"].unsqueeze(0).to(device)
        wt_esm_window_tokens = esm_features["wt_esm_window_tokens"].unsqueeze(0).to(device)
        mut_esm_window_tokens = esm_features["mut_esm_window_tokens"].unsqueeze(0).to(device)
        esm_window_padding_mask = esm_features["esm_window_padding_mask"].unsqueeze(0).to(device)
        esm_window_mutation_mask = esm_features["esm_window_mutation_mask"].unsqueeze(0).to(device)
        mutation_esm_embedding = None
    else:
        mutation_esm_embedding = None

    return {
        "antibody_seq": [wt_partner1],
        "antigen_seq": [wt_partner2],
        "mutant_antibody_seq": [mut_partner1],
        "mutant_antigen_seq": [mut_partner2],
        "foldx_energies": foldx_tensor,
        "foldx_features": foldx_features,
        "structure_data": structure_data,
        "wt_esm_embedding": wt_esm_embedding,
        "mut_esm_embedding": mut_esm_embedding,
        "mutation_esm_embedding": mutation_esm_embedding,
        "wt_esm_window_tokens": wt_esm_window_tokens,
        "mut_esm_window_tokens": mut_esm_window_tokens,
        "esm_window_padding_mask": esm_window_padding_mask,
        "esm_window_mutation_mask": esm_window_mutation_mask,
        "foldx_energy": foldx_energy,
        "foldx_mut_energy": mut_foldx_energy,
        "foldx_delta_interaction": mut_foldx_energy - foldx_energy,
        "foldx_buildmodel_ddg": foldx_buildmodel_ddg,
        "foldx_feature_mode": getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta"),
        "foldx_version": getattr(config, "FOLDX_VERSION", "foldx"),
        "foldx_path": os.path.abspath(config.FOLDX_PATH),
        "normalized_mutation": normalized_mutation,
        "structure_pdb_path": structure_pdb_path,
        "mutant_pdb_path": mutant_pdb_path,
    }


def infer_one(model, sample: Dict[str, object]) -> float:
    with torch.no_grad():
        pred = model(
            antibody_seqs=sample["antibody_seq"],
            antigen_seqs=sample["antigen_seq"],
            mutant_antibody_seqs=sample["mutant_antibody_seq"],
            mutant_antigen_seqs=sample["mutant_antigen_seq"],
            foldx_energies=sample["foldx_energies"],
            structure_data=sample["structure_data"],
            wt_esm_embedding=sample["wt_esm_embedding"],
            mut_esm_embedding=sample["mut_esm_embedding"],
            mutation_esm_embedding=sample["mutation_esm_embedding"],
            wt_esm_window_tokens=sample["wt_esm_window_tokens"],
            mut_esm_window_tokens=sample["mut_esm_window_tokens"],
            esm_window_padding_mask=sample["esm_window_padding_mask"],
            esm_window_mutation_mask=sample["esm_window_mutation_mask"],
            foldx_features=sample.get("foldx_features"),
        )
    return float(pred.squeeze().detach().cpu().item())


def main():
    args = parse_args()
    device = config.DEVICE

    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config.BASE_DIR, ckpt_path)

    model = build_model()
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

    rows = read_metadata(args.case_csv)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    output_rows: List[Dict[str, object]] = []
    for row in tqdm(rows, desc="Running inference", unit="case"):
        try:
            sample = prepare_model_inputs(
                row,
                foldx_processor,
                foldx_builder,
                esm_helper,
                device,
                strict_wt_check=not args.skip_wt_aa_check,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to prepare case-study sample "
                f"case_id={row.get('case_id', '')}, "
                f"mutation={row.get('mutation', '')}, "
                f"pdb_path={row.get('pdb_path', '')}"
            ) from exc
        pred_ddg = infer_one(model, sample)

        out_row = dict(row)
        out_row["pred_ddg"] = pred_ddg
        out_row["foldx_energy"] = sample["foldx_energy"]
        out_row["foldx_mut_energy"] = sample["foldx_mut_energy"]
        out_row["foldx_delta_interaction"] = sample["foldx_delta_interaction"]
        out_row["foldx_feature_mode"] = sample["foldx_feature_mode"]
        out_row["foldx_version"] = sample["foldx_version"]
        out_row["foldx_path"] = sample["foldx_path"]
        out_row["normalized_mutation"] = sample["normalized_mutation"]
        out_row["structure_pdb_path"] = sample["structure_pdb_path"]
        out_row["mutant_pdb_path"] = sample["mutant_pdb_path"]
        output_rows.append(out_row)

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
