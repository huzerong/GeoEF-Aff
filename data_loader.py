import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from collections import Counter
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from typing import Dict, List, Optional, Any
import joblib
import json

from foldx_processor import FoldXProcessor
from protein_features import AA1_TO_INDEX, ProteinFeatureExtractor


import config
from esm_local_tokens import (
    LOCAL_ESM_KEYS,
    LOCAL_TOKEN_VERSION_KEY,
    MUTATION_MASK_KEY,
    MUT_WINDOW_KEY,
    PADDING_MASK_KEY,
    POSITIONS_KEY,
    WT_WINDOW_KEY,
)


def build_all_chain_mapping(
    partner1_chains,
    partner2_chains,
) -> Dict[str, str]:
    """Map every partner chain while preserving the three role segment types."""
    mapping = {}
    for index, chain_id in enumerate(partner1_chains):
        if index == 0:
            role = "heavy"
        elif index == 1:
            role = "light"
        else:
            role = f"antibody_{index}"
        mapping[role] = str(chain_id)
    for index, chain_id in enumerate(partner2_chains):
        role = "antigen" if index == 0 else f"antigen_{index}"
        mapping[role] = str(chain_id)
    if not mapping:
        raise ValueError("Cannot build an empty structure chain mapping.")
    return mapping


def attach_structure_chain_indices(
    structure_data,
    chain_offset=0,
):
    atom_count = int(structure_data["coords"].shape[0])
    chain_ids = structure_data.get("chain_ids")
    if not isinstance(chain_ids, (list, tuple)) or len(chain_ids) != atom_count:
        raise ValueError(
            "Structure chain_ids must align with coords to build chain indices."
        )
    mapping = {}
    numeric_ids = []
    for chain_id in chain_ids:
        chain_key = str(chain_id)
        if chain_key not in mapping:
            mapping[chain_key] = int(chain_offset) + len(mapping)
        numeric_ids.append(mapping[chain_key])
    structure_data = dict(structure_data)
    structure_data["chain_numeric_ids"] = torch.tensor(
        numeric_ids,
        dtype=torch.long,
    )
    structure_data["structure_chain_mapping_version"] = getattr(
        config,
        "STRUCTURE_CHAIN_MAPPING_VERSION",
        2,
    )
    return structure_data, len(mapping)


def load_mutation_foldx_features(row_idx: int) -> Dict[str, Any]:
    cache_dir = getattr(
        config,
        "MUTATION_FOLDX_CACHE_DIR",
        os.path.join(config.BASE_DIR, "foldx_mutation_cache"),
    )
    cache_path = os.path.join(cache_dir, "sample_json", f"{row_idx}.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing mutation FoldX cache for row {row_idx}: {cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "ok":
        raise ValueError(
            f"Mutation FoldX cache for row {row_idx} is not ok: "
            f"status={data.get('status')!r}, path={cache_path}"
        )

    required = ("wt_energy", "mut_energy", "foldx_delta_interaction")
    missing = [key for key in required if data.get(key) is None]
    if missing:
        raise KeyError(f"Mutation FoldX cache for row {row_idx} is missing {missing}: {cache_path}")

    wt_energy = float(data["wt_energy"])
    mut_energy = float(data["mut_energy"])
    delta_interaction = float(data["foldx_delta_interaction"])
    calculated_delta = mut_energy - wt_energy
    if abs(delta_interaction - calculated_delta) > 1e-4:
        raise ValueError(
            f"Mutation FoldX delta mismatch for row {row_idx}: "
            f"cached={delta_interaction}, calculated={calculated_delta}, path={cache_path}"
        )

    return {
        "foldx_features": torch.tensor(
            [wt_energy, mut_energy, delta_interaction],
            dtype=torch.float32,
        ),
        "foldx_energy": torch.tensor(wt_energy, dtype=torch.float32),
        "foldx_mutant_pdb_path": data.get("mutant_pdb_path", ""),
        "foldx_repaired_pdb_path": data.get("repaired_pdb_path", ""),
    }

# 设置 joblib 缓存
memory = joblib.Memory(config.DISK_CACHE_DIR, verbose=0)

@memory.cache
def cached_structure_extraction(pdb_path: str, chain_mapping: Dict[str, str], use_atom_features: bool) -> Dict[str, torch.Tensor]:
    """
    Cached version of structure feature extraction.
    Creates a temporary extractor instance to avoid pickling the whole dataset/model.
    """
    extractor = ProteinFeatureExtractor(use_atom_features=use_atom_features)
    return extractor.extract_from_pdb(pdb_path, chain_mapping)


def ensure_structure_residue_ids(
    structure_features: Dict[str, Any],
    pdb_path: str,
    chain_mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Add PDB residue ids to cached structure features that were produced by older code."""
    structure_features = dict(structure_features)
    coords = structure_features.get("coords")
    if coords is None:
        return structure_features

    num_atoms = int(coords.shape[0])
    residue_ids = structure_features.get("residue_ids")
    if residue_ids is not None and len(residue_ids) == num_atoms:
        return structure_features

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    extractor = ProteinFeatureExtractor()
    rebuilt_residue_ids = []
    atoms_to_extract = ["CA", "C", "N", "O"]

    for chain_id in chain_mapping.values():
        if chain_id not in [chain.id for chain in structure.get_chains()]:
            continue
        chain = structure[0][chain_id]
        for residue in chain.get_residues():
            if not extractor._is_standard_residue(residue):
                continue
            residue_atom_count = sum(1 for atom_name in atoms_to_extract if atom_name in residue)
            if residue_atom_count < 3:
                continue
            emitted_atom_count = max(residue_atom_count, 4)
            residue_id = str(residue.get_id()[1]) + residue.get_id()[2].strip()
            rebuilt_residue_ids.extend([residue_id] * emitted_atom_count)

    if len(rebuilt_residue_ids) != num_atoms:
        raise RuntimeError(
            f"Could not recover residue_ids for cached structure {pdb_path}: "
            f"rebuilt={len(rebuilt_residue_ids)}, atoms={num_atoms}"
        )

    structure_features["residue_ids"] = rebuilt_residue_ids
    return structure_features


def attach_mutation_masks(
    structure_features: Dict[str, Any],
    mutation_sites: List[tuple],
    mutation_types: Optional[Dict[tuple, tuple]] = None,
    strict_wt_check: Optional[bool] = None,
) -> Dict[str, Any]:
    """Add mutation masks and the mutant residue type aligned to every structure atom."""
    structure_features = dict(structure_features)
    coords = structure_features.get("coords")
    if coords is None:
        return structure_features

    num_atoms = int(coords.shape[0])
    mutation_mask = torch.zeros(num_atoms, dtype=torch.bool)
    mutation_ca_mask = torch.zeros(num_atoms, dtype=torch.bool)
    aa_types = structure_features.get("aa_types")
    if not isinstance(aa_types, torch.Tensor) or int(aa_types.shape[0]) != num_atoms:
        raise ValueError("Structure aa_types must align with coords before attaching mutation types.")
    wt_aa_types = aa_types.clone().long()
    mutant_aa_types = aa_types.clone().long()
    mismatched_wt_sites = set()
    matched_sites = set()

    normalized_types = {}
    for site, substitution in (mutation_types or {}).items():
        chain_id, residue_id = site
        wt_aa, mut_aa = substitution
        wt_aa = str(wt_aa).upper()
        mut_aa = str(mut_aa).upper()
        if wt_aa not in AA1_TO_INDEX or wt_aa == "X":
            raise ValueError(f"Unsupported WT amino-acid type {wt_aa!r} at {site}.")
        if mut_aa not in AA1_TO_INDEX or mut_aa == "X":
            raise ValueError(f"Unsupported mutant amino-acid type {mut_aa!r} at {site}.")
        normalized_types[(str(chain_id), str(residue_id))] = (wt_aa, mut_aa)

    if mutation_sites and not normalized_types and getattr(config, "REQUIRE_MUTATION_TYPE_FEATURES", True):
        raise ValueError("Mutation sites were provided without WT/Mut amino-acid types.")

    if num_atoms == 0 or not mutation_sites:
        structure_features["mutation_mask"] = mutation_mask
        structure_features["mutation_ca_mask"] = mutation_ca_mask
        structure_features["wt_aa_types"] = wt_aa_types
        structure_features["mutant_aa_types"] = mutant_aa_types
        structure_features["mutation_site_count"] = 0
        structure_features["matched_mutation_site_count"] = 0
        structure_features["mutation_wt_mismatch_count"] = 0
        structure_features["mutation_type_feature_version"] = getattr(
            config, "MUTATION_TYPE_FEATURE_VERSION", 2
        )
        return structure_features

    site_set = {(str(chain_id), str(residue_id)) for chain_id, residue_id in mutation_sites}
    chain_ids = structure_features.get("chain_ids", [])
    residue_ids = structure_features.get("residue_ids", [])
    atom_names = structure_features.get("atom_names", [])
    seq_idx = structure_features.get("seq_idx")

    if len(chain_ids) != num_atoms:
        raise ValueError("Structure chain_ids must align with coords.")
    if len(residue_ids) != num_atoms and seq_idx is None:
        raise ValueError(
            "Structure needs aligned residue_ids or seq_idx for mutation matching."
        )

    seq_idx_list = seq_idx.detach().cpu().tolist() if isinstance(seq_idx, torch.Tensor) else seq_idx
    for atom_idx in range(num_atoms):
        if len(residue_ids) == num_atoms:
            residue_key = str(residue_ids[atom_idx])
        else:
            residue_key = str(seq_idx_list[atom_idx])
        site_key = (str(chain_ids[atom_idx]), residue_key)
        if site_key not in site_set:
            continue
        if site_key not in normalized_types:
            raise ValueError(f"Missing mutation type for structure site {site_key}.")
        matched_sites.add(site_key)
        mutation_mask[atom_idx] = True
        wt_aa, mut_aa = normalized_types[site_key]
        wt_idx = AA1_TO_INDEX[wt_aa]
        mut_idx = AA1_TO_INDEX[mut_aa]
        observed_idx = int(aa_types[atom_idx].item())
        if observed_idx != AA1_TO_INDEX["X"] and observed_idx != wt_idx:
            mismatched_wt_sites.add(site_key)
        wt_aa_types[atom_idx] = wt_idx
        mutant_aa_types[atom_idx] = mut_idx
        atom_name = atom_names[atom_idx] if atom_idx < len(atom_names) else ""
        if str(atom_name).strip().upper() == "CA":
            mutation_ca_mask[atom_idx] = True

    missing_sites = site_set.difference(matched_sites)
    if missing_sites:
        raise ValueError(
            "Mutation sites did not match structure residues: "
            f"{sorted(missing_sites)}"
        )

    if mutation_mask.any() and not mutation_ca_mask.any():
        mutation_ca_mask = mutation_mask.clone()

    structure_features["mutation_mask"] = mutation_mask
    structure_features["mutation_ca_mask"] = mutation_ca_mask
    structure_features["wt_aa_types"] = wt_aa_types
    structure_features["mutant_aa_types"] = mutant_aa_types
    structure_features["mutation_site_count"] = len(site_set)
    structure_features["matched_mutation_site_count"] = len(matched_sites)
    structure_features["mutation_wt_mismatch_count"] = len(mismatched_wt_sites)
    structure_features["mutation_type_feature_version"] = getattr(
        config, "MUTATION_TYPE_FEATURE_VERSION", 2
    )
    if strict_wt_check is None:
        strict_wt_check = getattr(config, "STRICT_MUTATION_WT_CHECK", True)
    if mismatched_wt_sites and strict_wt_check:
        raise ValueError(
            "WT amino-acid annotation disagrees with the structure at "
            f"{len(mismatched_wt_sites)} site(s): {sorted(mismatched_wt_sites)}"
        )
    return structure_features


def build_mutation_site_id(complex_id: str, mutation_string: str) -> str:
    sites = []
    for raw_token in str(mutation_string).split(","):
        token = raw_token.strip()
        if len(token) < 4:
            raise ValueError(f"Invalid mutation token for site grouping: {token!r}")
        sites.append(token[1:-1])
    return f"{str(complex_id).strip()}|{','.join(sorted(sites))}"


class SKEMPI2Dataset(Dataset):

    def __init__(
        self, 
        csv_path: str, 
        pdb_dir: str, 
        foldx_processor: FoldXProcessor,
        use_structure: bool = True,
        structure_feature_extractor: Optional[ProteinFeatureExtractor] = None
    ):
        self.df = pd.read_csv(csv_path, sep=";")
        self.df["ddG"] = (
            (8.314 / 4184)
            * (273.15 + 25.0)
            * (
                np.log(self.df["Affinity_mut_parsed"])
                - np.log(self.df["Affinity_wt_parsed"])
            )
        )
        self.df = self.df.dropna(subset=["ddG", "#Pdb", "Mutation(s)_cleaned"])
        
        # 如果启用了 USE_ONLY_CACHED_FOLDX，过滤掉没有缓存的样本
        if config.ENABLE_FOLDX and getattr(config, 'USE_ONLY_CACHED_FOLDX', False):
            print("WARNING: USE_ONLY_CACHED_FOLDX is enabled. Filtering dataset...")
            valid_indices = []
            for idx in self.df.index:
                # 检查缓存文件是否存在
                cache_path = os.path.join(config.FOLDX_CACHE_DIR, f"{idx}.pkl")
                if os.path.exists(cache_path):
                    try:
                        import pickle
                        with open(cache_path, "rb") as f:
                            data = pickle.load(f)
                        if isinstance(data, dict) and "foldx_energy" in data:
                            valid_indices.append(idx)
                    except Exception as e:
                        raise RuntimeError(f"Failed to inspect FoldX cache file {cache_path}") from e
            
            original_len = len(self.df)
            self.df = self.df.loc[valid_indices]
            print(f"Filtered dataset: {original_len} -> {len(self.df)} samples (FoldX Cached)")
            
            if len(self.df) == 0:
                print("Error: No cached FoldX data found! Training will fail.")

        self.pdb_dir = pdb_dir
        self.foldx_processor = foldx_processor
        self.parser = PDBParser(QUIET=True)
        
        self.use_structure = use_structure
        self.structure_feature_extractor = structure_feature_extractor or ProteinFeatureExtractor()
        
        self._structure_cache = {}

        # self._repaired_map = None
        # self._failed_ids = set()

    # def set_repaired_map(self, repaired_map, failed_ids=None):
    #     self._repaired_map = dict(repaired_map)
    #     self._failed_ids = set(failed_ids or [])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print("==========")
        # 使用真实的行索引来访问 DataFrame，防止索引错乱
        # 因为在 __init__ 中我们可能根据缓存过滤了 self.df
        row = self.df.iloc[idx]
        real_idx = self.df.index[idx] # 获取原始 CSV 中的索引
        
        try:
            pdb_id, group1, group2 = row["#Pdb"].split("_")
            pdb_path = os.path.abspath(os.path.join(self.pdb_dir, f"{pdb_id}.pdb"))

            if not os.path.exists(pdb_path):
                raise FileNotFoundError(f"PDB file not found for row {real_idx}: {pdb_path}")
            
            # --- FOLDX ENERGY LOADING (EARLY EXIT) ---
            # 如果启用了 USE_ONLY_CACHED_FOLDX，这里先检查缓存
            # 如果没有缓存，直接返回 None，避免后续昂贵的结构解析
            if not config.ENABLE_FOLDX:
                raise RuntimeError("ENABLE_FOLDX must be true for 3-feature retraining.")
            loaded_from_cache = False
            repaired_pdb_path = pdb_path # 默认使用原始路径，如果 FoldX 修复了则替换
            mutation_foldx = load_mutation_foldx_features(real_idx)
            foldx_features = None
            
            if config.ENABLE_FOLDX:
                cache_path = os.path.join(config.FOLDX_CACHE_DIR, f"{real_idx}.pkl")
                loaded_from_cache = False
                
                interaction_energy = float(mutation_foldx["foldx_energy"].item())
                foldx_features = mutation_foldx["foldx_features"]
                loaded_from_cache = True
                
                if not loaded_from_cache:
                    # 如果启用了 USE_ONLY_CACHED_FOLDX，这里应该直接跳过或返回 None
                    # 但为了鲁棒性，如果配置允许实时计算才计算
                    if getattr(config, 'USE_ONLY_CACHED_FOLDX', False):
                         # 如果强制只用缓存，但缓存没命中（可能是刚生成的还没刷盘，或者索引对不上），
                         # 这里我们不应该进行昂贵的计算。
                         # 我们可以返回 None 跳过这个样本，或者给个默认值。
                         # 为了不阻塞 dataloader，我们给个 0.0 并打印警告
                         # print(f"Warning: Cache miss for {real_idx} in strict mode.")
                         raise RuntimeError(f"Missing required mutation FoldX features for row {real_idx}")
                    else:
                        # 对于未命中的，尝试实时计算
                        # 但我们发现实时计算会导致 dataloader 卡死或超时，尤其是在多进程环境下
                        # 所以这里我们直接返回 0.0，不进行计算
                        raise RuntimeError(f"Missing required mutation FoldX features for row {real_idx}")

            # --- STRUCTURE PARSING ---
            structure = self.parser.get_structure(pdb_id, pdb_path)
            if structure is None:
                raise RuntimeError(f"Failed to parse structure for row {real_idx}: {pdb_path}")
            model = structure[0]

            seqs = {}
            res_id_to_idx = {}  # Map (chain_id, pdb_res_id) -> sequence_index
            
            for chain in model:
                residues = [res for res in chain.get_residues() if is_aa(res)]
                if residues:
                    seq_str = "".join([seq1(res.get_resname()) for res in residues])
                    seqs[chain.id] = seq_str
                    
                    # Build mapping for this chain
                    for idx_in_seq, res in enumerate(residues):
                        res_key = (chain.id, str(res.get_id()[1]) + res.get_id()[2].strip())
                        res_id_to_idx[res_key] = idx_in_seq

            muts = row["Mutation(s)_cleaned"].split(",")
            mut_chain = muts[0][1]

            if mut_chain in group1:
                ab_chains_ids, ag_chains_ids = list(group1), list(group2)
            else:
                ab_chains_ids, ag_chains_ids = list(group2), list(group1)

            # Construct WT sequences
            antibody_seq = "".join([seqs[c] for c in ab_chains_ids if c in seqs])
            antigen_seq = "".join([seqs[c] for c in ag_chains_ids if c in seqs])

            if not antibody_seq or not antigen_seq:
                raise ValueError(
                    f"Failed to build antibody/antigen sequences for row {real_idx}: "
                    f"pdb={pdb_id}, ab_chains={ab_chains_ids}, ag_chains={ag_chains_ids}"
                )
            
            # Construct Mutant sequences
            mutant_seqs_dict = seqs.copy()
            mutation_sites = []
            mutation_types = {}
            
            for mut in muts:
                orig_aa = mut[0].upper()
                chain_id = mut[1]
                new_aa = mut[-1].upper()
                res_num_str = mut[2:-1] # e.g. "38" or "38A"
                
                if chain_id in mutant_seqs_dict:
                    # Find the index
                    res_key = (chain_id, res_num_str)
                    if res_key in res_id_to_idx:
                        idx_in_chain = res_id_to_idx[res_key]
                        mutation_sites.append((chain_id, res_num_str))
                        mutation_types[(chain_id, res_num_str)] = (orig_aa, new_aa)
                        chain_seq = list(mutant_seqs_dict[chain_id])
                        chain_seq[idx_in_chain] = new_aa
                        mutant_seqs_dict[chain_id] = "".join(chain_seq)
                    else:
                        raise KeyError(f"Mutation residue {chain_id}{res_num_str} not found for row {real_idx}")
                else:
                    raise KeyError(f"Mutation chain {chain_id} not found for row {real_idx}")

            mutant_antibody_seq = "".join([mutant_seqs_dict[c] for c in ab_chains_ids if c in mutant_seqs_dict])
            mutant_antigen_seq = "".join([mutant_seqs_dict[c] for c in ag_chains_ids if c in mutant_seqs_dict])

            # repaired_pdb_path = self.foldx_processor.preprocess_pdb(pdb_path)
            # repaired_pdb_path = pdb_path # Moved up
            
            # --- FOLDX ENERGY LOADING ---
            # Already handled above
            if not loaded_from_cache and not getattr(config, 'USE_ONLY_CACHED_FOLDX', False):
                 # Try real-time calculation if not restricted
                 # try:
                        #     interaction_energy = self.foldx_processor.extract_features(
                        #         repaired_pdb_path,
                        #         partner1_chains=ab_chains_ids,
                        #         partner2_chains=ag_chains_ids,
                        #     )
                        # except Exception:
                        #     interaction_energy = 0.0
                        raise RuntimeError(f"Missing required mutation FoldX features for row {real_idx}")


            structure_data = None
            if self.use_structure:
                try:
                    # 使用 PDB ID 作为缓存键的一部分，避免重复解析
                    structure_data = self._extract_structure_features(
                        repaired_pdb_path, 
                        ab_chains_ids, 
                        ag_chains_ids,
                        pdb_id,
                        mutation_sites,
                        mutation_types,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to extract structure features for row {real_idx}, pdb={pdb_id}"
                    ) from e

            if self.use_structure and structure_data is None:
                raise RuntimeError(f"Structure features are required but missing for row {real_idx}, pdb={pdb_id}")
            
            delta_g = torch.tensor(row["ddG"], dtype=torch.float32)

            sample = {
                "antibody_seq": antibody_seq,
                "antigen_seq": antigen_seq,
                "mutant_antibody_seq": mutant_antibody_seq,
                "mutant_antigen_seq": mutant_antigen_seq,
                "foldx_energy": torch.tensor(interaction_energy, dtype=torch.float32),
                "delta_g": delta_g,
                "structure_data": structure_data,
                "pdb_id": pdb_id,
                "complex_id": str(row["#Pdb"]),
                "mutation_site_id": build_mutation_site_id(
                    str(row["#Pdb"]),
                    str(row["Mutation(s)_cleaned"]),
                ),
            }
            if foldx_features is None:
                raise RuntimeError(f"foldx_features are required but missing for row {real_idx}")
            sample["foldx_features"] = foldx_features
            sample["foldx_feature_mode"] = getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta")
            sample["foldx_mutant_pdb_path"] = mutation_foldx.get("foldx_mutant_pdb_path", "")
            sample["foldx_repaired_pdb_path"] = mutation_foldx.get("foldx_repaired_pdb_path", "")
            return sample
        except Exception as e:
            raise RuntimeError(f"Error loading dataset item idx={idx}, real_idx={real_idx}") from e
    
    def _extract_structure_features(
        self, 
        pdb_path: str, 
        ab_chains_ids: List[str], 
        ag_chains_ids: List[str],
        pdb_id: str,
        mutation_sites: Optional[List[tuple]] = None,
        mutation_types: Optional[Dict[tuple, tuple]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        # 使用 joblib 缓存的函数
        # 构造 chain_mapping
        chain_mapping = build_all_chain_mapping(
            ab_chains_ids,
            ag_chains_ids,
        )
            
        try:
            # 调用缓存函数
            structure_features = cached_structure_extraction(
                pdb_path, 
                chain_mapping, 
                self.structure_feature_extractor.use_atom_features
            )
            structure_features = ensure_structure_residue_ids(
                structure_features,
                pdb_path,
                chain_mapping,
            )
            
            if len(structure_features['coords']) == 0:
                raise ValueError(f"No structure atoms extracted from {pdb_path}")
            
            # 为批处理添加 batch_ids
            structure_features = attach_mutation_masks(
                structure_features,
                mutation_sites or [],
                mutation_types=mutation_types,
            )
            structure_features, _ = attach_structure_chain_indices(
                structure_features
            )
            structure_features['batch_ids'] = torch.zeros_like(structure_features['segment_ids'])
            
            return structure_features
            
        except Exception as e:
            raise RuntimeError(f"Error extracting structure features from {pdb_path}") from e


def _slice_atom_aligned_structure_data(structure_data, indices, atom_count):
    sliced = {}
    index_list = indices.detach().cpu().tolist()
    for key, value in structure_data.items():
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.shape[0] == atom_count
        ):
            sliced[key] = value[indices]
        elif isinstance(value, list) and len(value) == atom_count:
            sliced[key] = [value[index] for index in index_list]
        elif isinstance(value, tuple) and len(value) == atom_count:
            sliced[key] = tuple(value[index] for index in index_list)
        else:
            sliced[key] = value
    return sliced


def build_structure_residue_indices(structure_data, residue_offset=0):
    atom_count = int(structure_data["coords"].shape[0])
    atom_names = structure_data.get("atom_names")
    if not isinstance(atom_names, (list, tuple)) or len(atom_names) != atom_count:
        raise ValueError(
            "Structure atom_names must align with coords to build CA masks."
        )
    ca_mask = torch.tensor(
        [
            str(atom_name).strip().upper() == "CA"
            for atom_name in atom_names
        ],
        dtype=torch.bool,
    )

    chain_ids = structure_data.get("chain_ids")
    if not isinstance(chain_ids, (list, tuple)) or len(chain_ids) != atom_count:
        raise ValueError(
            "Structure chain_ids must align with coords to build residue IDs."
        )

    residue_mapping = {}
    residue_uids = []
    for chain_id, residue_seq_idx in zip(
        chain_ids,
        structure_data["seq_idx"].detach().cpu().tolist(),
    ):
        residue_key = (str(chain_id), int(residue_seq_idx))
        if residue_key not in residue_mapping:
            residue_mapping[residue_key] = (
                int(residue_offset) + len(residue_mapping)
            )
        residue_uids.append(residue_mapping[residue_key])
    return (
        ca_mask,
        torch.tensor(residue_uids, dtype=torch.long),
        len(residue_mapping),
    )


def _select_complete_residue_atoms(
    structure_data,
    max_atoms,
    deterministic=False,
):
    atom_count = int(structure_data["coords"].shape[0])
    if atom_count <= max_atoms:
        return torch.arange(atom_count, dtype=torch.long)

    segment_ids = structure_data["segment_ids"].detach().cpu().long()
    seq_idx = structure_data["seq_idx"].detach().cpu().long()
    mutation_mask = structure_data.get("mutation_mask")
    if not isinstance(mutation_mask, torch.Tensor):
        mutation_mask = torch.zeros(atom_count, dtype=torch.bool)
    else:
        mutation_mask = mutation_mask.detach().cpu().bool()

    residue_groups = {}
    residue_order = []
    for atom_index, key in enumerate(zip(segment_ids.tolist(), seq_idx.tolist())):
        if key not in residue_groups:
            residue_groups[key] = []
            residue_order.append(key)
        residue_groups[key].append(atom_index)

    required_keys = [
        key
        for key in residue_order
        if bool(mutation_mask[residue_groups[key]].any())
    ]
    required_atom_count = sum(len(residue_groups[key]) for key in required_keys)
    if required_atom_count > max_atoms:
        raise ValueError(
            "Mutation residues alone exceed MAX_STRUCTURE_ATOMS: "
            f"{required_atom_count} > {max_atoms}."
        )

    required_set = set(required_keys)
    candidate_keys = [
        key for key in residue_order if key not in required_set
    ]
    if not deterministic and candidate_keys:
        order = torch.randperm(len(candidate_keys)).tolist()
        candidate_keys = [candidate_keys[index] for index in order]

    selected_keys = list(required_keys)
    selected_atom_count = required_atom_count
    for key in candidate_keys:
        group_size = len(residue_groups[key])
        if selected_atom_count + group_size > max_atoms:
            continue
        selected_keys.append(key)
        selected_atom_count += group_size

    selected_indices = sorted(
        atom_index
        for key in selected_keys
        for atom_index in residue_groups[key]
    )
    if not selected_indices:
        raise ValueError("Residue-aware structure sampling selected no atoms.")
    return torch.tensor(selected_indices, dtype=torch.long)


def filter_none_collate(batch):
    none_count = sum(1 for b in batch if b is None)
    if none_count:
        raise ValueError(f"Received {none_count} None samples in a batch; failing fast.")
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("Received an empty batch; failing fast.")

    # 限制每个样本的结构原子数，避免 OOM
    max_atoms = getattr(config, 'MAX_STRUCTURE_ATOMS', None)
    deterministic_structure_sampling = getattr(config, 'DETERMINISTIC_STRUCTURE_SAMPLING', False)

    structure_data_list = []
    
    # 先提取序列并编码为tensor
    seq_length = 1000  # 最大序列长度
    encoded_batch = []
    
    for item in batch:
        sd = item.pop('structure_data', None)
        # 原子数超限时随机采样
        if sd is not None and max_atoms is not None and sd['coords'].shape[0] > max_atoms:
            n = sd['coords'].shape[0]
            idx = _select_complete_residue_atoms(
                sd,
                max_atoms=max_atoms,
                deterministic=deterministic_structure_sampling,
            )
            sd = _slice_atom_aligned_structure_data(sd, idx, n)
        structure_data_list.append(sd)
        
        # 编码序列为tensor
        encoded_item = {}
        for key in ['antibody_seq', 'antigen_seq', 'mutant_antibody_seq', 'mutant_antigen_seq']:
            if key in item:
                seq = item[key]
                if isinstance(seq, str):
                    encoded = [ord(c) for c in seq[:seq_length]]
                    encoded += [0] * (seq_length - len(encoded))
                    encoded_item[key] = torch.tensor(encoded, dtype=torch.long)
                else:
                    encoded_item[key] = seq
                del item[key]  # 删除原始字符串
        
        # 复制其他字段
        if "wt_esm_embedding" in item and "mutation_esm_embedding" not in item:
            wt_emb = item["wt_esm_embedding"]
            if isinstance(wt_emb, torch.Tensor):
                item["mutation_esm_embedding"] = torch.zeros(
                    wt_emb.numel() * 4,
                    dtype=wt_emb.dtype,
                )
        for k, v in item.items():
            encoded_item[k] = v
        
        encoded_batch.append(encoded_item)
    
    collated_batch = default_collate(encoded_batch)
    
    collated_structure_data = collate_structure_data(structure_data_list)
    if collated_structure_data is not None:
        collated_batch['structure_data'] = collated_structure_data
    else:
        raise ValueError("Structure data collation returned None; failing fast.")
    
    return collated_batch


def collate_structure_data(structure_data_list: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    valid_structures = [s for s in structure_data_list if s is not None]
    
    if not valid_structures:
        raise ValueError("No valid structures in batch.")
    
    batch_size = len(structure_data_list)
    
    if len(valid_structures) < batch_size:
        raise ValueError(
            f"Missing structure data for {batch_size - len(valid_structures)} "
            f"of {batch_size} samples."
        )
    
    collated = {}
    
    all_coords = []
    all_aa_types = []
    all_wt_aa_types = []
    all_mutant_aa_types = []
    all_atom_types = []
    all_segment_ids = []
    all_seq_idx = []
    all_batch_ids = []
    all_chain_numeric_ids = []
    all_ca_masks = []
    all_residue_uids = []
    all_mutation_masks = []
    all_mutation_ca_masks = []
    all_mutation_site_counts = []
    all_matched_mutation_site_counts = []
    all_mutation_wt_mismatch_counts = []
    
    atom_offset = 0
    residue_offset = 0
    chain_offset = 0
    
    for batch_id, struct_data in enumerate(valid_structures):
        all_coords.append(struct_data['coords'])
        all_aa_types.append(struct_data['aa_types'])
        wt_aa_types = struct_data.get('wt_aa_types')
        mutant_aa_types = struct_data.get('mutant_aa_types')
        if not isinstance(wt_aa_types, torch.Tensor):
            raise ValueError("Structure data is missing wt_aa_types.")
        if not isinstance(mutant_aa_types, torch.Tensor):
            raise ValueError("Structure data is missing mutant_aa_types.")
        if wt_aa_types.shape != struct_data['aa_types'].shape:
            raise ValueError("wt_aa_types must have the same shape as aa_types.")
        if mutant_aa_types.shape != struct_data['aa_types'].shape:
            raise ValueError("mutant_aa_types must have the same shape as aa_types.")
        all_wt_aa_types.append(wt_aa_types)
        all_mutant_aa_types.append(mutant_aa_types)
        all_atom_types.append(struct_data['atom_types'])
        all_segment_ids.append(struct_data['segment_ids'])
        chain_numeric_ids = struct_data.get("chain_numeric_ids")
        if not isinstance(chain_numeric_ids, torch.Tensor):
            raise ValueError("Structure data is missing chain_numeric_ids.")
        if chain_numeric_ids.shape != struct_data["segment_ids"].shape:
            raise ValueError(
                "chain_numeric_ids must have the same shape as segment_ids."
            )
        local_chain_ids = torch.unique(chain_numeric_ids, sorted=True)
        remapped_chain_ids = torch.empty_like(chain_numeric_ids)
        for local_index, chain_id in enumerate(local_chain_ids.tolist()):
            remapped_chain_ids[chain_numeric_ids == chain_id] = (
                chain_offset + local_index
            )
        all_chain_numeric_ids.append(remapped_chain_ids)
        chain_offset += int(local_chain_ids.numel())
        num_atoms = struct_data['coords'].shape[0]
        ca_mask, residue_uids, local_residue_count = (
            build_structure_residue_indices(
                struct_data,
                residue_offset=residue_offset,
            )
        )
        all_ca_masks.append(ca_mask)
        all_residue_uids.append(residue_uids)
        residue_offset += local_residue_count
        all_mutation_masks.append(
            struct_data.get('mutation_mask', torch.zeros(num_atoms, dtype=torch.bool))
        )
        all_mutation_ca_masks.append(
            struct_data.get('mutation_ca_mask', torch.zeros(num_atoms, dtype=torch.bool))
        )
        mutation_site_count = struct_data.get('mutation_site_count')
        matched_mutation_site_count = struct_data.get('matched_mutation_site_count')
        if mutation_site_count is None or matched_mutation_site_count is None:
            raise ValueError("Structure data is missing mutation site audit counts.")
        all_mutation_site_counts.append(int(mutation_site_count))
        all_matched_mutation_site_counts.append(int(matched_mutation_site_count))
        mismatch_count = struct_data.get('mutation_wt_mismatch_count')
        if mismatch_count is None:
            raise ValueError("Structure data is missing mutation_wt_mismatch_count.")
        all_mutation_wt_mismatch_counts.append(int(mismatch_count))
        
        seq_idx_adjusted = struct_data['seq_idx'] + atom_offset
        all_seq_idx.append(seq_idx_adjusted)
        
        batch_ids = torch.full_like(struct_data['segment_ids'], batch_id)
        all_batch_ids.append(batch_ids)
        
        atom_offset += len(struct_data['coords'])
    
    collated['coords'] = torch.cat(all_coords, dim=0)
    collated['aa_types'] = torch.cat(all_aa_types, dim=0)
    collated['wt_aa_types'] = torch.cat(all_wt_aa_types, dim=0)
    collated['mutant_aa_types'] = torch.cat(all_mutant_aa_types, dim=0)
    collated['atom_types'] = torch.cat(all_atom_types, dim=0)
    collated['segment_ids'] = torch.cat(all_segment_ids, dim=0)
    collated['chain_numeric_ids'] = torch.cat(
        all_chain_numeric_ids,
        dim=0,
    )
    collated['seq_idx'] = torch.cat(all_seq_idx, dim=0)
    collated['batch_ids'] = torch.cat(all_batch_ids, dim=0)
    collated['ca_mask'] = torch.cat(all_ca_masks, dim=0)
    collated['residue_uid'] = torch.cat(all_residue_uids, dim=0)
    collated['mutation_mask'] = torch.cat(all_mutation_masks, dim=0)
    collated['mutation_ca_mask'] = torch.cat(all_mutation_ca_masks, dim=0)
    collated['mutation_site_count'] = torch.tensor(
        all_mutation_site_counts,
        dtype=torch.long,
    )
    collated['matched_mutation_site_count'] = torch.tensor(
        all_matched_mutation_site_counts,
        dtype=torch.long,
    )
    collated['mutation_wt_mismatch_count'] = torch.tensor(
        all_mutation_wt_mismatch_counts,
        dtype=torch.long,
    )

    return collated


class PrecomputedDataset(Dataset):
    """从预计算的 .pt 文件快速加载样本，避免运行时 PDB 解析"""

    def __init__(self, precomputed_dir: str, validate_current_config: bool = True):
        self.precomputed_dir = precomputed_dir
        self.indices = []
        skipped = 0
        stale_reasons = Counter()
        for fname in os.listdir(precomputed_dir):
            if not fname.endswith(".pt"):
                continue
            sample_idx = int(fname.replace(".pt", ""))
            sample_path = os.path.join(precomputed_dir, fname)
            if validate_current_config:
                stale_reason = self._sample_stale_reason(sample_path)
                if stale_reason is not None:
                    stale_reasons[stale_reason] += 1
                    skipped += 1
                    continue
            self.indices.append(sample_idx)
        self.indices.sort()
        print(f"PrecomputedDataset: loaded {len(self.indices)} samples from {precomputed_dir}")
        if skipped:
            reason_text = ", ".join(f"{reason}={count}" for reason, count in stale_reasons.most_common())
            raise RuntimeError(
                f"PrecomputedDataset found {skipped} stale samples "
                f"({reason_text}); rerun this folder's precompute scripts to refresh them."
            )

    def _sample_stale_reason(self, sample_path: str) -> Optional[str]:
        try:
            sample = torch.load(sample_path, weights_only=False)
        except Exception:
            return "load_error"

        if sample.get("foldx_version") != getattr(config, "FOLDX_VERSION", "foldx"):
            return "foldx_version"
        if sample.get("foldx_path") != os.path.abspath(config.FOLDX_PATH):
            return "foldx_path"

        if getattr(config, "USE_STRUCTURE_FEATURES", True):
            structure_data = sample.get("structure_data")
            if structure_data is None:
                return "missing_structure"
            if "mutation_mask" not in structure_data or "mutation_ca_mask" not in structure_data:
                return "missing_mutation_mask"
            if "residue_ids" not in structure_data:
                return "missing_residue_ids"
            if "wt_aa_types" not in structure_data or "mutant_aa_types" not in structure_data:
                return "missing_mutation_aa_types"
            if (
                "mutation_site_count" not in structure_data
                or "matched_mutation_site_count" not in structure_data
            ):
                return "missing_mutation_site_audit_counts"
            if "mutation_wt_mismatch_count" not in structure_data:
                return "missing_mutation_wt_mismatch_count"
            if structure_data.get("mutation_type_feature_version") != getattr(
                config, "MUTATION_TYPE_FEATURE_VERSION", 2
            ):
                return "mutation_type_feature_version"
            chain_numeric_ids = structure_data.get("chain_numeric_ids")
            if not isinstance(chain_numeric_ids, torch.Tensor):
                return "missing_chain_numeric_ids"
            if chain_numeric_ids.shape != structure_data["segment_ids"].shape:
                return "bad_chain_numeric_ids_shape"
            if structure_data.get(
                "structure_chain_mapping_version"
            ) != getattr(config, "STRUCTURE_CHAIN_MAPPING_VERSION", 2):
                return "structure_chain_mapping_version"

        if not sample.get("complex_id"):
            return "missing_complex_id"
        if not sample.get("mutation_site_id"):
            return "missing_mutation_site_id"

        if getattr(config, "USE_PRECOMPUTED_ESM", False):
            required = {
                "wt_esm_embedding",
                "mut_esm_embedding",
                "mutation_esm_embedding",
            } | LOCAL_ESM_KEYS
            missing = required.difference(sample.keys())
            if missing:
                return "missing_" + "_".join(sorted(missing))
            if sample.get("esm_mutation_window_radius") != getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8):
                return "esm_window_radius"
            if sample.get(LOCAL_TOKEN_VERSION_KEY) != getattr(
                config,
                "ESM_LOCAL_TOKEN_VERSION",
                1,
            ):
                return "esm_local_token_version"
            max_tokens = getattr(config, "ESM_LOCAL_MAX_TOKENS", 32)
            expected_shapes = {
                "wt_esm_embedding": (1280,),
                "mut_esm_embedding": (1280,),
                "mutation_esm_embedding": (1280 * 4,),
                WT_WINDOW_KEY: (max_tokens, 1280),
                MUT_WINDOW_KEY: (max_tokens, 1280),
                PADDING_MASK_KEY: (max_tokens,),
                MUTATION_MASK_KEY: (max_tokens,),
                POSITIONS_KEY: (max_tokens,),
            }
            for key, expected_shape in expected_shapes.items():
                value = sample.get(key)
                if not isinstance(value, torch.Tensor):
                    return f"bad_{key}_type"
                if tuple(value.shape) != expected_shape:
                    return f"bad_{key}_shape"
                if value.is_floating_point() and not bool(
                    torch.isfinite(value).all()
                ):
                    return f"nonfinite_{key}"
            padding_mask = sample[PADDING_MASK_KEY]
            mutation_mask = sample[MUTATION_MASK_KEY]
            positions = sample[POSITIONS_KEY]
            if padding_mask.dtype != torch.bool:
                return "bad_esm_padding_mask_dtype"
            if mutation_mask.dtype != torch.bool:
                return "bad_esm_mutation_mask_dtype"
            if positions.dtype != torch.long:
                return "bad_esm_window_positions_dtype"
            if int((~padding_mask).sum().item()) < 1:
                return "empty_esm_local_window"
            if bool(mutation_mask[padding_mask].any()):
                return "mutation_on_padded_esm_token"
            if bool((positions[padding_mask] != -1).any()):
                return "position_on_padded_esm_token"

        foldx_features = sample.get("foldx_features")
        if foldx_features is None:
            return "missing_foldx_features"
        if not torch.is_tensor(foldx_features):
            return "bad_foldx_features_type"
        if int(foldx_features.numel()) != getattr(config, "FOLDX_FEATURE_DIM", 3):
            return "bad_foldx_features_dim"
        if sample.get("foldx_feature_mode") != getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta"):
            return "foldx_feature_mode"

        return None

    def _sample_matches_current_config(self, sample_path: str) -> bool:
        return self._sample_stale_reason(sample_path) is None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path = os.path.join(self.precomputed_dir, f"{real_idx}.pt")
        try:
            return torch.load(path, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load precomputed sample {path}") from e
