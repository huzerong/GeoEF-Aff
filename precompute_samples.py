"""
预计算所有样本特征并保存为.pt
训练时dataloader直接加载避免重复解析PDB
"""
import os
import argparse
import torch
import pandas as pd
import numpy as np
import pickle
import json
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from data_loader import attach_mutation_masks
from protein_features import ProteinFeatureExtractor

PRECOMPUTED_DIR = config.PRECOMPUTED_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute 3-feature FoldX training samples."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("PRECOMPUTE_WORKERS", "4")),
        help="Number of worker processes. Lower this if the host has limited RAM/file descriptors.",
    )
    return parser.parse_args()


def load_mutation_foldx_features(row_idx):
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


def foldx_cache_matches_current(data):
    configured_foldx_version = getattr(config, "FOLDX_VERSION", "foldx")
    if data.get("foldx_version") != configured_foldx_version:
        return False
    if data.get("foldx_path") and data.get("foldx_path") != os.path.abspath(config.FOLDX_PATH):
        return False
    return True


def is_known_foldx_failure(idx, foldx_cache_dir):
    cache_path = os.path.join(foldx_cache_dir, f"{idx}.pkl")
    if not os.path.exists(cache_path):
        return False
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return bool(data.get("foldx_failed")) and foldx_cache_matches_current(data)
    except Exception:
        return False


def process_one_sample(args):
    idx, row_dict, pdb_dir, foldx_cache_dir, use_structure, use_atom_features, out_path = args
    try:
        parser = PDBParser(QUIET=True)
        pdb_id, group1, group2 = row_dict["#Pdb"].split("_")
        pdb_path = os.path.abspath(os.path.join(pdb_dir, f"{pdb_id}.pdb"))

        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file not found for row {idx}: {pdb_path}")

        # FoldX energy
        mutation_foldx = load_mutation_foldx_features(idx)
        interaction_energy = float(mutation_foldx["foldx_energy"].item())
        foldx_features = mutation_foldx["foldx_features"]

        # Parse structure
        structure = parser.get_structure(pdb_id, pdb_path)
        if structure is None:
            raise RuntimeError(f"Failed to parse structure for row {idx}: {pdb_path}")
        model = structure[0]

        seqs = {}
        res_id_to_idx = {}
        for chain in model:
            residues = [res for res in chain.get_residues() if is_aa(res)]
            if residues:
                seq_str = "".join([seq1(res.get_resname()) for res in residues])
                seqs[chain.id] = seq_str
                for idx_in_seq, res in enumerate(residues):
                    res_key = (chain.id, str(res.get_id()[1]) + res.get_id()[2].strip())
                    res_id_to_idx[res_key] = idx_in_seq

        muts = row_dict["Mutation(s)_cleaned"].split(",")
        mut_chain = muts[0][1]

        if mut_chain in group1:
            ab_chains_ids, ag_chains_ids = list(group1), list(group2)
        else:
            ab_chains_ids, ag_chains_ids = list(group2), list(group1)

        antibody_seq = "".join([seqs[c] for c in ab_chains_ids if c in seqs])
        antigen_seq = "".join([seqs[c] for c in ag_chains_ids if c in seqs])
        if not antibody_seq or not antigen_seq:
            raise ValueError(
                f"Failed to build antibody/antigen sequences for row {idx}: "
                f"pdb={pdb_id}, ab_chains={ab_chains_ids}, ag_chains={ag_chains_ids}"
            )

        # Mutant sequences
        mutant_seqs_dict = seqs.copy()
        mutation_sites = []
        for mut in muts:
            chain_id = mut[1]
            new_aa = mut[-1]
            res_num_str = mut[2:-1]
            if chain_id in mutant_seqs_dict:
                res_key = (chain_id, res_num_str)
                if res_key in res_id_to_idx:
                    idx_in_chain = res_id_to_idx[res_key]
                    mutation_sites.append((chain_id, idx_in_chain))
                    chain_seq = list(mutant_seqs_dict[chain_id])
                    chain_seq[idx_in_chain] = new_aa
                    mutant_seqs_dict[chain_id] = "".join(chain_seq)
                else:
                    raise KeyError(f"Mutation residue {chain_id}{res_num_str} not found for row {idx}")
            else:
                raise KeyError(f"Mutation chain {chain_id} not found for row {idx}")

        mutant_antibody_seq = "".join([mutant_seqs_dict[c] for c in ab_chains_ids if c in mutant_seqs_dict])
        mutant_antigen_seq = "".join([mutant_seqs_dict[c] for c in ag_chains_ids if c in mutant_seqs_dict])

        # Structure features
        structure_data = None
        if use_structure:
            try:
                extractor = ProteinFeatureExtractor(use_atom_features=use_atom_features)
                chain_mapping = {}
                if len(ab_chains_ids) >= 1:
                    chain_mapping['heavy'] = ab_chains_ids[0]
                if len(ab_chains_ids) >= 2:
                    chain_mapping['light'] = ab_chains_ids[1]
                if len(ag_chains_ids) >= 1:
                    chain_mapping['antigen'] = ag_chains_ids[0]

                sf = extractor.extract_from_pdb(pdb_path, chain_mapping)
                if len(sf['coords']) > 0:
                    sf = attach_mutation_masks(sf, mutation_sites)
                    sf['batch_ids'] = torch.zeros_like(sf['segment_ids'])
                    structure_data = sf
            except Exception as e:
                raise RuntimeError(f"Failed to extract structure features for row {idx}, pdb={pdb_id}") from e

        if use_structure and structure_data is None:
            raise RuntimeError(f"Structure features are required but missing for row {idx}, pdb={pdb_id}")

        ddG = (8.314 / 4184) * (273.15 + 25.0) * (
            np.log(row_dict["Affinity_mut_parsed"]) - np.log(row_dict["Affinity_wt_parsed"])
        )

        sample = {
            "antibody_seq": antibody_seq,
            "antigen_seq": antigen_seq,
            "mutant_antibody_seq": mutant_antibody_seq,
            "mutant_antigen_seq": mutant_antigen_seq,
            "foldx_energy": torch.tensor(interaction_energy, dtype=torch.float32),
            "delta_g": torch.tensor(ddG, dtype=torch.float32),
            "structure_data": structure_data,
            "pdb_id": pdb_id,
            "foldx_version": getattr(config, "FOLDX_VERSION", "foldx"),
            "foldx_path": os.path.abspath(config.FOLDX_PATH),
        }
        if foldx_features is None:
            raise RuntimeError(f"foldx_features are required but missing for row {idx}")
        sample["foldx_features"] = foldx_features
        sample["foldx_feature_mode"] = getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta")
        sample["foldx_mutant_pdb_path"] = mutation_foldx.get("foldx_mutant_pdb_path", "")
        sample["foldx_repaired_pdb_path"] = mutation_foldx.get("foldx_repaired_pdb_path", "")
        torch.save(sample, out_path)
        return idx, out_path

    except Exception as e:
        raise RuntimeError(f"Failed to precompute sample row {idx}") from e


def main():
    args = parse_args()
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

    df = pd.read_csv(config.CSV_PATH, sep=";")
    df = df.dropna(subset=["#Pdb", "Mutation(s)_cleaned", "Affinity_mut_parsed", "Affinity_wt_parsed"])
    if "original_row_idx" not in df.columns:
        df["original_row_idx"] = df.index.astype(int)

    # Check which samples already exist
    todo = []
    skipped_foldx_failed = 0
    for i in range(len(df)):
        out_path = os.path.join(PRECOMPUTED_DIR, f"{i}.pt")
        needs_update = True
        if os.path.exists(out_path):
            try:
                sample = torch.load(out_path, weights_only=False)
                structure_data = sample.get("structure_data")
                has_mutation_masks = (
                    structure_data is not None
                    and "mutation_mask" in structure_data
                    and "mutation_ca_mask" in structure_data
                )
                needs_update = (
                    sample.get("foldx_version") != getattr(config, "FOLDX_VERSION", "foldx")
                    or sample.get("foldx_path") != os.path.abspath(config.FOLDX_PATH)
                    or (config.USE_STRUCTURE_FEATURES and not has_mutation_masks)
                    or sample.get("foldx_feature_mode") != getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta")
                )
                if getattr(config, "USE_MUTATION_FOLDX_FEATURES", True):
                    foldx_features = sample.get("foldx_features")
                    has_three_foldx_features = (
                        isinstance(foldx_features, torch.Tensor)
                        and int(foldx_features.numel()) == getattr(config, "FOLDX_FEATURE_DIM", 3)
                    )
                    mutation_cache_path = os.path.join(
                        getattr(config, "MUTATION_FOLDX_CACHE_DIR", os.path.join(config.BASE_DIR, "foldx_mutation_cache")),
                        "sample_json",
                        f"{int(df.iloc[i]['original_row_idx'])}.json",
                    )
                    if os.path.exists(mutation_cache_path) and not has_three_foldx_features:
                        needs_update = True
            except Exception:
                needs_update = True
        if needs_update:
            row = df.iloc[i]
            real_idx = int(row["original_row_idx"])
            out_path = os.path.join(PRECOMPUTED_DIR, f"{i}.pt")
            todo.append((real_idx, row.to_dict(), config.PDB_DIR, config.FOLDX_CACHE_DIR,
                          config.USE_STRUCTURE_FEATURES, config.USE_ATOM_FEATURES, out_path))

    print(f"Total samples: {len(df)}, to precompute: {len(todo)}")
    if skipped_foldx_failed:
        print(f"Skipped known FoldX failures: {skipped_foldx_failed}")
    if not todo:
        print("All samples already precomputed!")
        return

    success = 0

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(process_one_sample, t): t[0] for t in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Precomputing"):
            real_idx = futures[future]
            idx, saved_path = future.result()
            if int(idx) != int(real_idx):
                raise RuntimeError(f"Worker row mismatch: expected {real_idx}, got {idx}")
            if not os.path.exists(saved_path):
                raise FileNotFoundError(f"Worker reported success but output is missing: {saved_path}")
            success += 1

    print(f"Done. Success: {success}")


if __name__ == "__main__":
    main()
