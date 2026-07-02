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
from data_loader import attach_mutation_masks, cached_structure_extraction
from eval_pth_metrics import build_model, load_weights
from foldx_processor import FoldXProcessor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


CASE_FOLDX_FEATURE_CACHE_VERSION = 3


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
        data = [("complex", partner1_seq + partner2_seq)]
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
        _, wt_tokens, wt_len = self.encode_complex_tokens(wt_partner1, wt_partner2)
        _, mut_tokens, mut_len = self.encode_complex_tokens(mut_partner1, mut_partner2)
        max_len = min(wt_len, mut_len, len(wt_partner1 + wt_partner2), len(mut_partner1 + mut_partner2))
        mutation_positions = [
            idx for idx in range(max_len)
            if (wt_partner1 + wt_partner2)[idx] != (mut_partner1 + mut_partner2)[idx]
        ]
        if not mutation_positions:
            return wt_tokens.new_zeros(wt_tokens.shape[-1] * 4)

        pos_tensor = torch.tensor(mutation_positions, dtype=torch.long, device=wt_tokens.device)
        wt_site = wt_tokens[pos_tensor].mean(dim=0)
        mut_site = mut_tokens[pos_tensor].mean(dim=0)

        window_radius = getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8)
        window_mask = torch.zeros(max_len, dtype=torch.bool, device=wt_tokens.device)
        for pos in mutation_positions:
            start = max(0, pos - window_radius)
            end = min(max_len, pos + window_radius + 1)
            window_mask[start:end] = True
        window_idx = torch.nonzero(window_mask, as_tuple=True)[0]
        wt_window = wt_tokens[window_idx].mean(dim=0)
        mut_window = mut_tokens[window_idx].mean(dim=0)
        return torch.cat([wt_site, mut_site, mut_site - wt_site, mut_window - wt_window], dim=0).detach()


def parse_args():
    parser = argparse.ArgumentParser(description="Run case-study inference with best_model1.pth-compatible inputs.")
    parser.add_argument("--case-csv", required=True, help="Path to metadata CSV.")
    parser.add_argument("--ckpt", default="best_model1.pth", help="Checkpoint path.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path for predictions.")
    parser.add_argument("--task", choices=["rbd_ddg", "antibody_opt"], required=True)
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
    chain_mapping: Dict[str, str] = {}
    if len(group1) == 1:
        chain_mapping["heavy"] = group1[0]
    elif len(group1) >= 2:
        chain_mapping["heavy"] = group1[0]
        chain_mapping["light"] = group1[1]

    if len(group2) >= 1:
        chain_mapping["antigen"] = group2[0]

    return chain_mapping


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
    for token in normalized_mutation.split(","):
        _, chain_id, res_num_str, _ = parse_mutation_token(token, default_chain=default_chain)
        res_key = (chain_id, res_num_str)
        if res_key in res_id_to_idx:
            mutation_sites.append((chain_id, res_id_to_idx[res_key]))

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
    cached_foldx_features = load_case_foldx_feature_cache(
        pdb_path=pdb_path,
        chain_group_1=str(row["chain_group_1"]),
        chain_group_2=str(row["chain_group_2"]),
        normalized_mutation=normalized_mutation,
    )
    should_build_mutant_pdb = (
        getattr(config, "ENABLE_FOLDX", True)
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

    structure_data = extract_structure_data(structure_pdb_path, group1, group2, mutation_sites)
    if structure_data is not None:
        for key, value in structure_data.items():
            if isinstance(value, torch.Tensor):
                structure_data[key] = value.to(device)

    if cached_foldx_features is not None:
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
    if cached_foldx_features is None and mutant_pdb_path:
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
    if esm_helper is not None:
        wt_esm_embedding = esm_helper.encode_complex(wt_partner1, wt_partner2).unsqueeze(0).to(device)
        mut_esm_embedding = esm_helper.encode_complex(mut_partner1, mut_partner2).unsqueeze(0).to(device)
        mutation_esm_embedding = esm_helper.encode_mutation_context(
            wt_partner1,
            wt_partner2,
            mut_partner1,
            mut_partner2,
        ).unsqueeze(0).to(device)
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
        sample = prepare_model_inputs(row, foldx_processor, foldx_builder, esm_helper, device)
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
