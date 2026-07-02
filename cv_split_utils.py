"""Deterministic fold assignment helpers without ML dependencies."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


def assign_sample_folds(
    num_samples: int, num_folds: int = 3, seed: int = 42
) -> List[List[int]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    if num_samples < num_folds:
        raise ValueError(
            f"Only {num_samples} samples are available for {num_folds} folds."
        )

    rng = random.Random(seed)
    positions = list(range(num_samples))
    rng.shuffle(positions)

    folds: List[List[int]] = [[] for _ in range(num_folds)]
    for offset, position in enumerate(positions):
        folds[offset % num_folds].append(position)

    covered = sorted(position for fold in folds for position in fold)
    if covered != list(range(num_samples)):
        raise RuntimeError("Fold assignment does not cover every sample exactly once.")
    return [sorted(fold) for fold in folds]


def assign_pdb_group_folds(
    group_ids: Sequence[str], num_folds: int = 3, seed: int = 42
) -> Tuple[List[List[int]], List[List[str]]]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")
    group_to_positions: Dict[str, List[int]] = defaultdict(list)
    for position, raw_group in enumerate(group_ids):
        group = str(raw_group).strip()
        if not group:
            raise ValueError(f"Empty group id at position {position}.")
        group_to_positions[group].append(position)
    if len(group_to_positions) < num_folds:
        raise ValueError(
            f"Only {len(group_to_positions)} groups are available for {num_folds} folds."
        )

    rng = random.Random(seed)
    grouped_items = list(group_to_positions.items())
    rng.shuffle(grouped_items)
    grouped_items.sort(key=lambda item: len(item[1]), reverse=True)

    folds: List[List[int]] = [[] for _ in range(num_folds)]
    fold_groups: List[List[str]] = [[] for _ in range(num_folds)]
    for group, positions in grouped_items:
        target = min(
            range(num_folds),
            key=lambda index: (len(folds[index]), len(fold_groups[index]), index),
        )
        folds[target].extend(positions)
        fold_groups[target].append(group)

    covered = sorted(position for fold in folds for position in fold)
    if covered != list(range(len(group_ids))):
        raise RuntimeError("Fold assignment does not cover every sample exactly once.")
    for left in range(num_folds):
        for right in range(left + 1, num_folds):
            overlap = set(fold_groups[left]).intersection(fold_groups[right])
            if overlap:
                raise RuntimeError(f"Group leakage detected: {sorted(overlap)}")
    return [sorted(fold) for fold in folds], [sorted(groups) for groups in fold_groups]
