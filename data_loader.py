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
from protein_features import ProteinFeatureExtractor


import config


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


def attach_mutation_masks(
    structure_features: Dict[str, Any],
    mutation_sites: List[tuple],
) -> Dict[str, Any]:
    """Add atom-level masks for mutated residues to one extracted structure."""
    structure_features = dict(structure_features)
    coords = structure_features.get("coords")
    if coords is None:
        return structure_features

    num_atoms = int(coords.shape[0])
    mutation_mask = torch.zeros(num_atoms, dtype=torch.bool)
    mutation_ca_mask = torch.zeros(num_atoms, dtype=torch.bool)
    if num_atoms == 0 or not mutation_sites:
        structure_features["mutation_mask"] = mutation_mask
        structure_features["mutation_ca_mask"] = mutation_ca_mask
        return structure_features

    site_set = {(str(chain_id), int(seq_idx)) for chain_id, seq_idx in mutation_sites}
    chain_ids = structure_features.get("chain_ids", [])
    atom_names = structure_features.get("atom_names", [])
    seq_idx = structure_features.get("seq_idx")

    if seq_idx is None or len(chain_ids) != num_atoms:
        structure_features["mutation_mask"] = mutation_mask
        structure_features["mutation_ca_mask"] = mutation_ca_mask
        return structure_features

    seq_idx_list = seq_idx.detach().cpu().tolist() if isinstance(seq_idx, torch.Tensor) else seq_idx
    for atom_idx in range(num_atoms):
        site_key = (str(chain_ids[atom_idx]), int(seq_idx_list[atom_idx]))
        if site_key not in site_set:
            continue
        mutation_mask[atom_idx] = True
        atom_name = atom_names[atom_idx] if atom_idx < len(atom_names) else ""
        if str(atom_name).strip().upper() == "CA":
            mutation_ca_mask[atom_idx] = True

    if mutation_mask.any() and not mutation_ca_mask.any():
        mutation_ca_mask = mutation_mask.clone()

    structure_features["mutation_mask"] = mutation_mask
    structure_features["mutation_ca_mask"] = mutation_ca_mask
    return structure_features

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
            
            for mut in muts:
                orig_aa = mut[0]
                chain_id = mut[1]
                new_aa = mut[-1]
                res_num_str = mut[2:-1] # e.g. "38" or "38A"
                
                if chain_id in mutant_seqs_dict:
                    # Find the index
                    res_key = (chain_id, res_num_str)
                    if res_key in res_id_to_idx:
                        idx_in_chain = res_id_to_idx[res_key]
                        mutation_sites.append((chain_id, idx_in_chain))
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
    ) -> Optional[Dict[str, torch.Tensor]]:
        # 使用 joblib 缓存的函数
        # 构造 chain_mapping
        chain_mapping = {}
        
        if len(ab_chains_ids) == 1:
            chain_mapping['heavy'] = ab_chains_ids[0]
        elif len(ab_chains_ids) >= 2:
            chain_mapping['heavy'] = ab_chains_ids[0]
            chain_mapping['light'] = ab_chains_ids[1]
        
        if len(ag_chains_ids) >= 1:
            chain_mapping['antigen'] = ag_chains_ids[0]
            
        try:
            # 调用缓存函数
            structure_features = cached_structure_extraction(
                pdb_path, 
                chain_mapping, 
                self.structure_feature_extractor.use_atom_features
            )
            
            if len(structure_features['coords']) == 0:
                raise ValueError(f"No structure atoms extracted from {pdb_path}")
            
            # 为批处理添加 batch_ids
            structure_features = attach_mutation_masks(
                structure_features,
                mutation_sites or [],
            )
            structure_features['batch_ids'] = torch.zeros_like(structure_features['segment_ids'])
            
            return structure_features
            
        except Exception as e:
            raise RuntimeError(f"Error extracting structure features from {pdb_path}") from e


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
            mutation_mask = sd.get('mutation_mask')
            mutation_ca_mask = sd.get('mutation_ca_mask')
            keep_mask = torch.zeros(n, dtype=torch.bool)
            if isinstance(mutation_mask, torch.Tensor) and mutation_mask.shape[0] == n:
                keep_mask |= mutation_mask.cpu().bool()
            if isinstance(mutation_ca_mask, torch.Tensor) and mutation_ca_mask.shape[0] == n:
                keep_mask |= mutation_ca_mask.cpu().bool()
            keep_idx = torch.nonzero(keep_mask, as_tuple=True)[0]
            if keep_idx.numel() == 0:
                if deterministic_structure_sampling:
                    idx = torch.arange(n)[:max_atoms]
                else:
                    idx = torch.randperm(n)[:max_atoms].sort().values
            elif keep_idx.numel() >= max_atoms:
                if deterministic_structure_sampling:
                    idx = keep_idx[:max_atoms].sort().values
                else:
                    idx = keep_idx[torch.randperm(keep_idx.numel())[:max_atoms]].sort().values
            else:
                rest_idx = torch.nonzero(~keep_mask, as_tuple=True)[0]
                if deterministic_structure_sampling:
                    extra_idx = rest_idx[:max_atoms - keep_idx.numel()]
                else:
                    extra_idx = rest_idx[torch.randperm(rest_idx.numel())[:max_atoms - keep_idx.numel()]]
                idx = torch.cat([keep_idx, extra_idx]).sort().values
            sd = {k: v[idx] if isinstance(v, torch.Tensor) and v.shape[0] == n else v
                  for k, v in sd.items()}
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
    all_atom_types = []
    all_segment_ids = []
    all_seq_idx = []
    all_batch_ids = []
    all_mutation_masks = []
    all_mutation_ca_masks = []
    
    atom_offset = 0
    
    for batch_id, struct_data in enumerate(valid_structures):
        all_coords.append(struct_data['coords'])
        all_aa_types.append(struct_data['aa_types'])
        all_atom_types.append(struct_data['atom_types'])
        all_segment_ids.append(struct_data['segment_ids'])
        num_atoms = struct_data['coords'].shape[0]
        all_mutation_masks.append(
            struct_data.get('mutation_mask', torch.zeros(num_atoms, dtype=torch.bool))
        )
        all_mutation_ca_masks.append(
            struct_data.get('mutation_ca_mask', torch.zeros(num_atoms, dtype=torch.bool))
        )
        
        seq_idx_adjusted = struct_data['seq_idx'] + atom_offset
        all_seq_idx.append(seq_idx_adjusted)
        
        batch_ids = torch.full_like(struct_data['segment_ids'], batch_id)
        all_batch_ids.append(batch_ids)
        
        atom_offset += len(struct_data['coords'])
    
    collated['coords'] = torch.cat(all_coords, dim=0)
    collated['aa_types'] = torch.cat(all_aa_types, dim=0)
    collated['atom_types'] = torch.cat(all_atom_types, dim=0)
    collated['segment_ids'] = torch.cat(all_segment_ids, dim=0)
    collated['seq_idx'] = torch.cat(all_seq_idx, dim=0)
    collated['batch_ids'] = torch.cat(all_batch_ids, dim=0)
    collated['mutation_mask'] = torch.cat(all_mutation_masks, dim=0)
    collated['mutation_ca_mask'] = torch.cat(all_mutation_ca_masks, dim=0)

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

        if getattr(config, "USE_PRECOMPUTED_ESM", False):
            required = {"wt_esm_embedding", "mut_esm_embedding", "mutation_esm_embedding"}
            missing = required.difference(sample.keys())
            if missing:
                return "missing_" + "_".join(sorted(missing))
            if sample.get("esm_mutation_window_radius") != getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8):
                return "esm_window_radius"

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
