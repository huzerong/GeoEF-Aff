from typing import Dict, Iterable, List, Optional

import torch


WT_WINDOW_KEY = "wt_esm_window_tokens"
MUT_WINDOW_KEY = "mut_esm_window_tokens"
PADDING_MASK_KEY = "esm_window_padding_mask"
MUTATION_MASK_KEY = "esm_window_mutation_mask"
POSITIONS_KEY = "esm_window_positions"
LOCAL_TOKEN_VERSION_KEY = "esm_local_token_version"

LOCAL_ESM_KEYS = {
    WT_WINDOW_KEY,
    MUT_WINDOW_KEY,
    PADDING_MASK_KEY,
    MUTATION_MASK_KEY,
    POSITIONS_KEY,
    LOCAL_TOKEN_VERSION_KEY,
}


def find_substitution_positions(wt_sequence: str, mutant_sequence: str) -> List[int]:
    if len(wt_sequence) != len(mutant_sequence):
        raise ValueError(
            "Local ESM windows only support substitutions; WT and mutant "
            f"lengths differ ({len(wt_sequence)} != {len(mutant_sequence)})."
        )
    return [
        index
        for index, (wt_aa, mutant_aa) in enumerate(
            zip(wt_sequence, mutant_sequence)
        )
        if wt_aa != mutant_aa
    ]


def select_local_window_positions(
    mutation_positions: Iterable[int],
    sequence_length: int,
    radius: int = 8,
    max_tokens: int = 32,
) -> List[int]:
    mutation_positions = sorted({int(position) for position in mutation_positions})
    if not mutation_positions:
        raise ValueError("No substitution positions were found for local ESM tokens.")
    if sequence_length < 1:
        raise ValueError("Sequence length must be positive.")
    if radius < 0:
        raise ValueError("Window radius cannot be negative.")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")
    if mutation_positions[0] < 0 or mutation_positions[-1] >= sequence_length:
        raise IndexError(
            f"Mutation positions {mutation_positions} exceed sequence length "
            f"{sequence_length}."
        )
    if len(mutation_positions) > max_tokens:
        raise ValueError(
            f"{len(mutation_positions)} mutation positions exceed the "
            f"{max_tokens}-token local ESM limit."
        )

    union = set()
    for position in mutation_positions:
        start = max(0, position - radius)
        end = min(sequence_length, position + radius + 1)
        union.update(range(start, end))

    if len(union) <= max_tokens:
        return sorted(union)

    mutation_set = set(mutation_positions)
    candidates = union.difference(mutation_set)
    ordered_candidates = sorted(
        candidates,
        key=lambda position: (
            min(abs(position - mutation) for mutation in mutation_positions),
            position,
        ),
    )
    selected = mutation_positions + ordered_candidates[
        : max_tokens - len(mutation_positions)
    ]
    return sorted(selected)


def build_local_esm_context(
    wt_sequence: str,
    mutant_sequence: str,
    radius: int = 8,
    max_tokens: int = 32,
    max_context_length: int = 1022,
    expected_mutation_count: Optional[int] = None,
    mutation_positions: Optional[Iterable[int]] = None,
) -> Dict[str, object]:
    sequence_mutation_positions = find_substitution_positions(
        wt_sequence,
        mutant_sequence,
    )
    if mutation_positions is None:
        mutation_positions = sequence_mutation_positions
        if (
            expected_mutation_count is not None
            and len(mutation_positions) != int(expected_mutation_count)
        ):
            raise ValueError(
                "Sequence-derived mutation count does not match the annotation: "
                f"{len(mutation_positions)} != {int(expected_mutation_count)}."
            )
    else:
        mutation_positions = sorted({int(position) for position in mutation_positions})
        if (
            expected_mutation_count is not None
            and len(mutation_positions) != int(expected_mutation_count)
        ):
            raise ValueError(
                "Annotated mutation position count does not match the annotation: "
                f"{len(mutation_positions)} != {int(expected_mutation_count)}."
            )
        unannotated_sequence_changes = sorted(
            set(sequence_mutation_positions).difference(mutation_positions)
        )
        if unannotated_sequence_changes:
            raise ValueError(
                "Sequence-derived mutation positions are not covered by the "
                "annotation-derived positions: "
                f"{unannotated_sequence_changes} not in {mutation_positions}."
            )
    if any(position >= len(wt_sequence) or position < 0 for position in mutation_positions):
        raise ValueError(
            "Mutation positions exceed sequence bounds: "
            f"length={len(wt_sequence)}, mutation_positions={mutation_positions}."
        )
    selected_positions = select_local_window_positions(
        mutation_positions,
        sequence_length=len(wt_sequence),
        radius=radius,
        max_tokens=max_tokens,
    )
    if max_context_length < 1:
        raise ValueError("max_context_length must be positive.")

    selected_span = selected_positions[-1] - selected_positions[0] + 1
    if selected_span <= max_context_length:
        context_length = min(len(wt_sequence), max_context_length)
        spare_context = context_length - selected_span
        if selected_positions[-1] < max_context_length:
            # Reuse the standard global ESM pass whenever it contains the
            # complete requested window.
            context_start = 0
        else:
            context_start = selected_positions[0] - spare_context // 2
            context_start = max(
                0,
                min(context_start, len(wt_sequence) - context_length),
            )
        context_end = context_start + context_length
        context_token_indices = [
            position - context_start
            for position in selected_positions
        ]
        wt_context_sequence = wt_sequence[context_start:context_end]
        mutant_context_sequence = mutant_sequence[
            context_start:context_end
        ]
        context_mode = "contiguous_crop"
    else:
        # Extremely distant multi-site mutations cannot fit into one ESM input.
        # Keep the exact selected windows without dropping any mutation token.
        context_start = None
        context_token_indices = list(range(len(selected_positions)))
        wt_context_sequence = "".join(
            wt_sequence[position]
            for position in selected_positions
        )
        mutant_context_sequence = "".join(
            mutant_sequence[position]
            for position in selected_positions
        )
        context_mode = "selected_windows"

    return {
        "wt_context_sequence": wt_context_sequence,
        "mutant_context_sequence": mutant_context_sequence,
        "context_token_indices": context_token_indices,
        "selected_positions": selected_positions,
        "mutation_positions": mutation_positions,
        "context_start": context_start,
        "context_mode": context_mode,
    }


def pack_preselected_esm_tokens(
    wt_context_tokens: torch.Tensor,
    mutant_context_tokens: torch.Tensor,
    context_token_indices: Iterable[int],
    selected_positions: Iterable[int],
    mutation_positions: Iterable[int],
    max_tokens: int = 32,
) -> Dict[str, torch.Tensor]:
    if wt_context_tokens.dim() != 2 or mutant_context_tokens.dim() != 2:
        raise ValueError(
            "WT and mutant context tokens must have shape [length, dim]."
        )
    if wt_context_tokens.shape[1] != mutant_context_tokens.shape[1]:
        raise ValueError("WT and mutant ESM token dimensions differ.")

    context_token_indices = [int(index) for index in context_token_indices]
    selected_positions = [int(position) for position in selected_positions]
    mutation_positions = {int(position) for position in mutation_positions}
    if len(context_token_indices) != len(selected_positions):
        raise ValueError(
            "Context token indices and selected positions must align."
        )
    if not selected_positions:
        raise ValueError("No local ESM positions were selected.")
    if len(selected_positions) > max_tokens:
        raise ValueError(
            f"{len(selected_positions)} selected positions exceed "
            f"max_tokens={max_tokens}."
        )
    available_length = min(
        int(wt_context_tokens.shape[0]),
        int(mutant_context_tokens.shape[0]),
    )
    if (
        min(context_token_indices) < 0
        or max(context_token_indices) >= available_length
    ):
        raise ValueError(
            "Selected local positions exceed encoded ESM context length: "
            f"indices={context_token_indices}, "
            f"available_length={available_length}."
        )

    index = torch.tensor(
        context_token_indices,
        dtype=torch.long,
        device=wt_context_tokens.device,
    )
    valid_count = len(selected_positions)
    embedding_dim = int(wt_context_tokens.shape[1])
    packed_wt = wt_context_tokens.new_zeros((max_tokens, embedding_dim))
    packed_mutant = mutant_context_tokens.new_zeros(
        (max_tokens, embedding_dim)
    )
    packed_wt[:valid_count] = wt_context_tokens[index]
    packed_mutant[:valid_count] = mutant_context_tokens[index]

    padding_mask = torch.ones(
        max_tokens,
        dtype=torch.bool,
        device=wt_context_tokens.device,
    )
    padding_mask[:valid_count] = False
    mutation_mask = torch.zeros(
        max_tokens,
        dtype=torch.bool,
        device=wt_context_tokens.device,
    )
    mutation_mask[:valid_count] = torch.tensor(
        [
            position in mutation_positions
            for position in selected_positions
        ],
        dtype=torch.bool,
        device=wt_context_tokens.device,
    )
    positions = torch.full(
        (max_tokens,),
        -1,
        dtype=torch.long,
        device=wt_context_tokens.device,
    )
    positions[:valid_count] = torch.tensor(
        selected_positions,
        dtype=torch.long,
        device=wt_context_tokens.device,
    )
    return {
        WT_WINDOW_KEY: packed_wt,
        MUT_WINDOW_KEY: packed_mutant,
        PADDING_MASK_KEY: padding_mask,
        MUTATION_MASK_KEY: mutation_mask,
        POSITIONS_KEY: positions,
    }


def pool_packed_mutation_esm_features(
    packed: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build the legacy four-part mutation feature from valid local tokens."""
    wt_tokens = packed[WT_WINDOW_KEY]
    mutant_tokens = packed[MUT_WINDOW_KEY]
    valid_mask = ~packed[PADDING_MASK_KEY]
    mutation_mask = packed[MUTATION_MASK_KEY]
    if not bool(mutation_mask.any()):
        raise ValueError("Packed local ESM tokens contain no mutation token.")
    wt_site = wt_tokens[mutation_mask].mean(dim=0)
    mutant_site = mutant_tokens[mutation_mask].mean(dim=0)
    wt_window = wt_tokens[valid_mask].mean(dim=0)
    mutant_window = mutant_tokens[valid_mask].mean(dim=0)
    return torch.cat(
        [
            wt_site,
            mutant_site,
            mutant_site - wt_site,
            mutant_window - wt_window,
        ],
        dim=0,
    )


def packed_local_esm_metadata_matches(
    packed: Dict[str, object],
    wt_sequence: str,
    mutant_sequence: str,
    radius: int = 8,
    max_tokens: int = 32,
) -> bool:
    """Check cached positions and masks against the full, untruncated sequence."""
    try:
        mutation_positions = find_substitution_positions(
            wt_sequence,
            mutant_sequence,
        )
        selected_positions = select_local_window_positions(
            mutation_positions,
            sequence_length=len(wt_sequence),
            radius=radius,
            max_tokens=max_tokens,
        )
        expected_positions = selected_positions + [-1] * (
            max_tokens - len(selected_positions)
        )
        expected_padding = [False] * len(selected_positions) + [True] * (
            max_tokens - len(selected_positions)
        )
        mutation_set = set(mutation_positions)
        expected_mutation_mask = [
            position in mutation_set
            for position in selected_positions
        ] + [False] * (max_tokens - len(selected_positions))
        return (
            packed[POSITIONS_KEY].detach().cpu().tolist()
            == expected_positions
            and packed[PADDING_MASK_KEY].detach().cpu().tolist()
            == expected_padding
            and packed[MUTATION_MASK_KEY].detach().cpu().tolist()
            == expected_mutation_mask
        )
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return False


def pack_local_esm_tokens(
    wt_tokens: torch.Tensor,
    mutant_tokens: torch.Tensor,
    wt_sequence: str,
    mutant_sequence: str,
    radius: int = 8,
    max_tokens: int = 32,
    expected_mutation_count: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if wt_tokens.dim() != 2 or mutant_tokens.dim() != 2:
        raise ValueError("WT and mutant ESM tokens must have shape [length, dim].")
    if wt_tokens.shape[1] != mutant_tokens.shape[1]:
        raise ValueError("WT and mutant ESM token dimensions differ.")

    mutation_positions = find_substitution_positions(
        wt_sequence,
        mutant_sequence,
    )
    if (
        expected_mutation_count is not None
        and len(mutation_positions) != int(expected_mutation_count)
    ):
        raise ValueError(
            "Sequence-derived mutation count does not match the annotation: "
            f"{len(mutation_positions)} != {int(expected_mutation_count)}."
        )

    available_length = min(
        len(wt_sequence),
        int(wt_tokens.shape[0]),
        int(mutant_tokens.shape[0]),
    )
    if any(position >= available_length for position in mutation_positions):
        raise ValueError(
            "At least one mutation lies outside the cached ESM token range: "
            f"available_length={available_length}, "
            f"mutation_positions={mutation_positions}."
        )

    selected_positions = select_local_window_positions(
        mutation_positions,
        sequence_length=available_length,
        radius=radius,
        max_tokens=max_tokens,
    )
    return pack_preselected_esm_tokens(
        wt_context_tokens=wt_tokens,
        mutant_context_tokens=mutant_tokens,
        context_token_indices=selected_positions,
        selected_positions=selected_positions,
        mutation_positions=mutation_positions,
        max_tokens=max_tokens,
    )
