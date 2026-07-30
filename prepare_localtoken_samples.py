"""Clone reusable sample data while forcing local ESM token regeneration."""

import argparse
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

import config
from esm_local_tokens import LOCAL_ESM_KEYS


OLD_ESM_KEYS = {
    "wt_esm_embedding",
    "mut_esm_embedding",
    "mutation_esm_embedding",
    "esm_mutation_window_radius",
} | LOCAL_ESM_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone structure/FoldX sample payloads from the template cache and "
            "remove ESM fields so local multi-token embeddings are recomputed."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=config.SOURCE_PRECOMPUTED_DIR,
        help="Template precomputed-sample directory.",
    )
    parser.add_argument(
        "--destination-dir",
        default=config.PRECOMPUTED_DIR,
        help="New local-token sample directory.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def validate_reusable_sample(sample: Dict[str, object], path: str) -> None:
    required = {
        "antibody_seq",
        "antigen_seq",
        "mutant_antibody_seq",
        "mutant_antigen_seq",
        "delta_g",
        "foldx_features",
        "foldx_feature_mode",
        "structure_data",
        "complex_id",
        "mutation_site_id",
    }
    missing = sorted(required.difference(sample))
    if missing:
        raise ValueError(f"{path} is missing reusable fields: {missing}")

    foldx_features = sample["foldx_features"]
    if not isinstance(foldx_features, torch.Tensor):
        raise TypeError(f"{path}: foldx_features must be a tensor.")
    if int(foldx_features.numel()) != getattr(config, "FOLDX_FEATURE_DIM", 3):
        raise ValueError(
            f"{path}: expected {config.FOLDX_FEATURE_DIM} FoldX features, "
            f"got {int(foldx_features.numel())}."
        )
    if sample.get("foldx_feature_mode") != getattr(
        config,
        "FOLDX_FEATURE_MODE",
        "wt_mut_delta",
    ):
        raise ValueError(f"{path}: incompatible foldx_feature_mode.")

    structure_data = sample["structure_data"]
    if not isinstance(structure_data, dict):
        raise TypeError(f"{path}: structure_data must be a dictionary.")
    required_structure = {
        "coords",
        "aa_types",
        "atom_types",
        "segment_ids",
        "seq_idx",
        "atom_names",
        "chain_ids",
        "chain_numeric_ids",
        "mutation_mask",
        "mutation_ca_mask",
        "wt_aa_types",
        "mutant_aa_types",
        "mutation_site_count",
        "matched_mutation_site_count",
        "mutation_wt_mismatch_count",
    }
    missing_structure = sorted(required_structure.difference(structure_data))
    if missing_structure:
        raise ValueError(
            f"{path} is missing reusable structure fields: "
            f"{missing_structure}"
        )
    if structure_data.get("structure_chain_mapping_version") != getattr(
        config,
        "STRUCTURE_CHAIN_MAPPING_VERSION",
        2,
    ):
        raise ValueError(f"{path} has an incompatible structure chain mapping.")


def clone_without_esm(sample: Dict[str, object]) -> Dict[str, object]:
    cloned = dict(sample)
    for key in OLD_ESM_KEYS:
        cloned.pop(key, None)
    return cloned


def _pt_files(directory: str) -> List[str]:
    try:
        names = [
            entry.name
            for entry in os.scandir(directory)
            if entry.is_file() and entry.name.endswith(".pt")
        ]
    except OSError:
        return []
    try:
        return sorted(
            names,
            key=lambda name: int(os.path.splitext(name)[0]),
        )
    except ValueError:
        return sorted(names)


def _candidate_precomputed_dirs(
    project_root: str,
    destination_dir: str,
) -> List[str]:
    candidates = []
    try:
        experiment_dirs = [
            entry.path
            for entry in os.scandir(project_root)
            if entry.is_dir()
        ]
    except OSError:
        return candidates

    for experiment_dir in experiment_dirs:
        try:
            children = os.scandir(experiment_dir)
        except OSError:
            continue
        with children:
            for child in children:
                if (
                    child.is_dir()
                    and child.name.startswith("precomputed_samples")
                    and os.path.abspath(child.path) != destination_dir
                ):
                    candidates.append(os.path.abspath(child.path))
    return sorted(set(candidates))


def _compatible_source_score(directory: str) -> tuple:
    parent_name = os.path.basename(os.path.dirname(directory))
    cache_name = os.path.basename(directory)
    return (
        "beneficial_grouprank_muttype" in parent_name,
        "localtoken32" not in parent_name,
        "beneficial_grouprank_muttype" in cache_name,
    )


def discover_compatible_source(
    project_root: str,
    destination_dir: str,
) -> Optional[str]:
    compatible = []
    for candidate in _candidate_precomputed_dirs(
        project_root,
        destination_dir,
    ):
        files = _pt_files(candidate)
        if not files:
            continue
        sample_path = os.path.join(candidate, files[0])
        try:
            sample = torch.load(
                sample_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(sample, dict):
                continue
            validate_reusable_sample(sample, sample_path)
        except Exception:
            continue
        compatible.append(candidate)

    if not compatible:
        return None
    compatible.sort(
        key=lambda path: (_compatible_source_score(path), path),
        reverse=True,
    )
    best_score = _compatible_source_score(compatible[0])
    best = [
        path
        for path in compatible
        if _compatible_source_score(path) == best_score
    ]
    if len(best) > 1:
        options = "\n  ".join(best)
        raise RuntimeError(
            "Multiple equally compatible template caches were found. "
            "Choose one explicitly with --source-dir:\n  "
            f"{options}"
        )
    return compatible[0]


def main() -> None:
    args = parse_args()
    source_dir = os.path.abspath(args.source_dir)
    destination_dir = os.path.abspath(args.destination_dir)
    if source_dir == destination_dir:
        raise ValueError("Source and destination sample directories must differ.")
    if not os.path.isdir(source_dir) or not _pt_files(source_dir):
        configured_source = source_dir
        source_dir = discover_compatible_source(
            project_root=os.path.abspath(config.PROJECT_ROOT),
            destination_dir=destination_dir,
        )
        if source_dir is None:
            raise FileNotFoundError(
                "Configured template sample directory was not found and no "
                "compatible precomputed cache could be discovered.\n"
                f"Configured path: {configured_source}\n"
                "Build new sample payloads from the reusable structure/FoldX "
                "caches instead:\n"
                "  python precompute_samples.py\n"
                "  python precompute_esm_embeddings.py"
            )
        print(
            "Configured source cache was unavailable; automatically selected: "
            f"{source_dir}"
        )

    files = _pt_files(source_dir)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No .pt samples found in {source_dir}")

    os.makedirs(destination_dir, exist_ok=True)
    ready_marker = os.path.join(
        destination_dir,
        ".esm_localtoken32_ready.json",
    )
    if os.path.exists(ready_marker):
        os.remove(ready_marker)
    written = 0
    skipped = 0
    for name in tqdm(files, desc="Migrating reusable samples"):
        source_path = os.path.join(source_dir, name)
        destination_path = os.path.join(destination_dir, name)
        if os.path.exists(destination_path) and not args.overwrite:
            skipped += 1
            continue

        sample = torch.load(
            source_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(sample, dict):
            raise TypeError(f"{source_path}: expected a sample dictionary.")
        validate_reusable_sample(sample, source_path)
        migrated = clone_without_esm(sample)

        temporary_path = f"{destination_path}.tmp.{os.getpid()}"
        torch.save(migrated, temporary_path)
        os.replace(temporary_path, destination_path)
        written += 1

    print(f"Source      : {source_dir}")
    print(f"Destination : {destination_dir}")
    print(f"Written     : {written}")
    print(f"Skipped     : {skipped}")
    print("Next: python precompute_esm_embeddings.py")


if __name__ == "__main__":
    main()
