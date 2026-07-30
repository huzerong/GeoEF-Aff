"""
预计算ESM-2嵌入并写入已有.pt文件
训练时直接加载跳过forwardpass
"""
import json
import os
import torch
from esm.pretrained import load_model_and_alphabet
from tqdm import tqdm

import config
from esm_local_tokens import (
    LOCAL_ESM_KEYS,
    LOCAL_TOKEN_VERSION_KEY,
    MUTATION_MASK_KEY,
    MUT_WINDOW_KEY,
    PADDING_MASK_KEY,
    POSITIONS_KEY,
    WT_WINDOW_KEY,
    build_local_esm_context,
    packed_local_esm_metadata_matches,
    pack_preselected_esm_tokens,
    pool_packed_mutation_esm_features,
)

PRECOMPUTED_DIR = config.PRECOMPUTED_DIR
BATCH_SIZE = 8
ESM_EMBEDDING_KEYS = {
    "wt_esm_embedding",
    "mut_esm_embedding",
    "mutation_esm_embedding",
}
REQUIRED_ESM_KEYS = ESM_EMBEDDING_KEYS | LOCAL_ESM_KEYS
CACHE_READY_MARKER = os.path.join(
    PRECOMPUTED_DIR,
    ".esm_localtoken32_ready.json",
)
MAX_SEQ_LEN = 1022  # ESM2最大token, BOS/EOS = 1024


def encode_sequences(esm_model, alphabet, sequences, device, return_tokens=False):
    """返回mean-pooled嵌入"""
    sequences = [seq[:MAX_SEQ_LEN] for seq in sequences]
    batch_converter = alphabet.get_batch_converter()
    data = [("seq", seq) for seq in sequences]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        results = esm_model(
            batch_tokens,
            repr_layers=[esm_model.num_layers],
            return_contacts=False,
        )

    token_repr = results["representations"][esm_model.num_layers]
    seq_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    token_repr = token_repr[:, 1:-1]  # 去掉 BOS/EOS

    embeddings = []
    actual_lengths = []
    for i, slen in enumerate(seq_lens):
        actual_len = int(slen.item()) - 2
        actual_lengths.append(actual_len)
        embeddings.append(token_repr[i, :actual_len].mean(0).float().cpu())

    if return_tokens:
        return embeddings, token_repr.float().cpu(), actual_lengths
    return embeddings


def find_mutation_positions(wt_seq, mut_seq, max_len):
    limit = min(len(wt_seq), len(mut_seq), max_len)
    return [idx for idx in range(limit) if wt_seq[idx] != mut_seq[idx]]


def pool_mutation_esm_features(wt_tokens, mut_tokens, wt_lengths, mut_lengths, wt_seqs, mut_seqs):
    window_radius = getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8)
    embed_dim = wt_tokens.shape[-1]
    features = []

    for i, (wt_seq, mut_seq) in enumerate(zip(wt_seqs, mut_seqs)):
        max_len = min(wt_lengths[i], mut_lengths[i])
        mut_positions = find_mutation_positions(wt_seq, mut_seq, max_len)
        if not mut_positions:
            features.append(torch.zeros(embed_dim * 4, dtype=torch.float32))
            continue

        pos_tensor = torch.tensor(mut_positions, dtype=torch.long)
        wt_site = wt_tokens[i, pos_tensor].mean(dim=0)
        mut_site = mut_tokens[i, pos_tensor].mean(dim=0)

        window_mask = torch.zeros(max_len, dtype=torch.bool)
        for pos in mut_positions:
            start = max(0, pos - window_radius)
            end = min(max_len, pos + window_radius + 1)
            window_mask[start:end] = True
        window_idx = torch.nonzero(window_mask, as_tuple=True)[0]
        wt_window = wt_tokens[i, window_idx].mean(dim=0)
        mut_window = mut_tokens[i, window_idx].mean(dim=0)

        features.append(torch.cat([wt_site, mut_site, mut_site - wt_site, mut_window - wt_window], dim=0))

    return features


def sample_has_current_esm(sample):
    if not REQUIRED_ESM_KEYS.issubset(sample):
        return False
    if sample.get(LOCAL_TOKEN_VERSION_KEY) != getattr(
        config,
        "ESM_LOCAL_TOKEN_VERSION",
        1,
    ):
        return False
    if sample.get("esm_mutation_window_radius") != getattr(
        config,
        "ESM_MUTATION_WINDOW_RADIUS",
        8,
    ):
        return False

    max_tokens = getattr(config, "ESM_LOCAL_MAX_TOKENS", 32)
    expected_dim = 1280
    expected_shapes = {
        "wt_esm_embedding": (expected_dim,),
        "mut_esm_embedding": (expected_dim,),
        "mutation_esm_embedding": (expected_dim * 4,),
        WT_WINDOW_KEY: (max_tokens, expected_dim),
        MUT_WINDOW_KEY: (max_tokens, expected_dim),
        PADDING_MASK_KEY: (max_tokens,),
        MUTATION_MASK_KEY: (max_tokens,),
        POSITIONS_KEY: (max_tokens,),
    }
    for key, expected_shape in expected_shapes.items():
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            return False
        if tuple(value.shape) != expected_shape:
            return False
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            return False

    padding_mask = sample[PADDING_MASK_KEY]
    mutation_mask = sample[MUTATION_MASK_KEY]
    positions = sample[POSITIONS_KEY]
    if padding_mask.dtype != torch.bool or mutation_mask.dtype != torch.bool:
        return False
    if positions.dtype != torch.long:
        return False
    if bool(mutation_mask[padding_mask].any()):
        return False
    if bool((positions[padding_mask] != -1).any()):
        return False
    if int((~padding_mask).sum().item()) < 1:
        return False
    sequence_values = [
        sample.get("antibody_seq"),
        sample.get("antigen_seq"),
        sample.get("mutant_antibody_seq"),
        sample.get("mutant_antigen_seq"),
    ]
    if not all(isinstance(value, str) for value in sequence_values):
        return False
    wt_sequence = sequence_values[0] + sequence_values[1]
    mutant_sequence = sequence_values[2] + sequence_values[3]
    if not packed_local_esm_metadata_matches(
        sample,
        wt_sequence,
        mutant_sequence,
        radius=getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8),
        max_tokens=max_tokens,
    ):
        return False
    return True


def try_reuse_cached_esm(fname, sample):
    source_dir = getattr(config, "SOURCE_PRECOMPUTED_DIR", "")
    if not source_dir:
        return False
    source_path = os.path.join(source_dir, fname)
    destination_path = os.path.join(PRECOMPUTED_DIR, fname)
    if not os.path.isfile(source_path):
        return False
    if os.path.abspath(source_path) == os.path.abspath(destination_path):
        return False

    try:
        source = torch.load(source_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    sequence_keys = {
        "antibody_seq",
        "antigen_seq",
        "mutant_antibody_seq",
        "mutant_antigen_seq",
    }
    if any(source.get(key) != sample.get(key) for key in sequence_keys):
        return False
    if not sample_has_current_esm(source):
        return False
    if source.get("esm_mutation_window_radius") != getattr(
        config,
        "ESM_MUTATION_WINDOW_RADIUS",
        8,
    ):
        return False

    for key in REQUIRED_ESM_KEYS:
        value = source[key]
        sample[key] = (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
        )
    sample["esm_mutation_window_radius"] = getattr(
        config,
        "ESM_MUTATION_WINDOW_RADIUS",
        8,
    )
    torch.save(sample, destination_path)
    return True


def write_cache_ready_marker(all_files):
    payload = {
        "sample_count": len(all_files),
        "esm_local_token_version": getattr(
            config,
            "ESM_LOCAL_TOKEN_VERSION",
            1,
        ),
        "esm_local_max_tokens": getattr(
            config,
            "ESM_LOCAL_MAX_TOKENS",
            32,
        ),
        "esm_mutation_window_radius": getattr(
            config,
            "ESM_MUTATION_WINDOW_RADIUS",
            8,
        ),
    }
    temporary_path = f"{CACHE_READY_MARKER}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, CACHE_READY_MARKER)


def main():
    all_files = sorted(
        [f for f in os.listdir(PRECOMPUTED_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.replace(".pt", "")),
    )
    print(f"Total precomputed samples: {len(all_files)}")
    if not all_files:
        raise RuntimeError(
            f"No precomputed samples found in {PRECOMPUTED_DIR}. "
            "Run precompute_samples.py successfully before precomputing ESM embeddings."
        )

    #检查嵌入
    todo = []
    reused = 0
    for fname in all_files:
        path = os.path.join(PRECOMPUTED_DIR, fname)
        sample = torch.load(path, weights_only=False)
        foldx_features = sample.get("foldx_features")
        if not isinstance(foldx_features, torch.Tensor) or int(foldx_features.numel()) != getattr(config, "FOLDX_FEATURE_DIM", 3):
            raise ValueError(f"{path} is missing 3-dimensional foldx_features.")
        if sample.get("foldx_feature_mode") != getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta"):
            raise ValueError(f"{path} has stale foldx_feature_mode={sample.get('foldx_feature_mode')!r}.")
        if (
            not sample_has_current_esm(sample)
        ):
            if try_reuse_cached_esm(fname, sample):
                reused += 1
            else:
                todo.append((fname, sample))

    print(
        "Reused ESM embeddings from "
        f"{getattr(config, 'SOURCE_PRECOMPUTED_DIR', '')}: {reused}"
    )
    print(f"Samples needing ESM embeddings: {len(todo)}")
    if not todo:
        write_cache_ready_marker(all_files)
        print("All samples already have ESM embeddings!")
        return

    #加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {config.ESM_MODEL_NAME} on {device}...")
    esm_model, alphabet = load_model_and_alphabet(config.ESM_MODEL_NAME)
    esm_model = esm_model.to(device).eval()
    for p in esm_model.parameters():
        p.requires_grad = False
    print(f"ESM model loaded. Embed dim: {esm_model.embed_dim}")

    #批处理
    for start in tqdm(range(0, len(todo), BATCH_SIZE), desc="Precomputing ESM"):
        batch = todo[start : start + BATCH_SIZE]
        fnames = [b[0] for b in batch]
        samples = [b[1] for b in batch]

        # WT sequences: antibody + antigen
        wt_seqs = [s["antibody_seq"] + s["antigen_seq"] for s in samples]
        # Mutant sequences
        mut_seqs = [s["mutant_antibody_seq"] + s["mutant_antigen_seq"] for s in samples]

        wt_embs, wt_tokens, wt_lengths = encode_sequences(
            esm_model, alphabet, wt_seqs, device, return_tokens=True
        )
        mut_embs, mut_tokens, mut_lengths = encode_sequences(
            esm_model, alphabet, mut_seqs, device, return_tokens=True
        )
        mutation_embs = pool_mutation_esm_features(
            wt_tokens,
            mut_tokens,
            wt_lengths,
            mut_lengths,
            wt_seqs,
            mut_seqs,
        )
        local_contexts = []
        for i, sample in enumerate(samples):
            structure_data = sample.get("structure_data") or {}
            local_contexts.append(
                build_local_esm_context(
                    wt_sequence=wt_seqs[i],
                    mutant_sequence=mut_seqs[i],
                    radius=getattr(
                        config,
                        "ESM_MUTATION_WINDOW_RADIUS",
                        8,
                    ),
                    max_tokens=getattr(
                        config,
                        "ESM_LOCAL_MAX_TOKENS",
                        32,
                    ),
                    max_context_length=MAX_SEQ_LEN,
                    expected_mutation_count=structure_data.get(
                        "mutation_site_count"
                    ),
                )
            )

        local_wt_tokens = [None] * len(samples)
        local_mut_tokens = [None] * len(samples)
        crop_indices = []
        crop_wt_sequences = []
        crop_mut_sequences = []
        for i, context in enumerate(local_contexts):
            if context["context_start"] == 0:
                local_wt_tokens[i] = wt_tokens[i, : wt_lengths[i]]
                local_mut_tokens[i] = mut_tokens[i, : mut_lengths[i]]
            else:
                crop_indices.append(i)
                crop_wt_sequences.append(context["wt_context_sequence"])
                crop_mut_sequences.append(context["mutant_context_sequence"])

        if crop_indices:
            _, crop_wt_tokens, crop_wt_lengths = encode_sequences(
                esm_model,
                alphabet,
                crop_wt_sequences,
                device,
                return_tokens=True,
            )
            _, crop_mut_tokens, crop_mut_lengths = encode_sequences(
                esm_model,
                alphabet,
                crop_mut_sequences,
                device,
                return_tokens=True,
            )
            for crop_idx, sample_idx in enumerate(crop_indices):
                local_wt_tokens[sample_idx] = crop_wt_tokens[
                    crop_idx,
                    : crop_wt_lengths[crop_idx],
                ]
                local_mut_tokens[sample_idx] = crop_mut_tokens[
                    crop_idx,
                    : crop_mut_lengths[crop_idx],
                ]

        local_esm = []
        for i, _sample in enumerate(samples):
            context = local_contexts[i]
            local_esm.append(
                pack_preselected_esm_tokens(
                    wt_context_tokens=local_wt_tokens[i],
                    mutant_context_tokens=local_mut_tokens[i],
                    context_token_indices=context["context_token_indices"],
                    selected_positions=context["selected_positions"],
                    mutation_positions=context["mutation_positions"],
                    max_tokens=getattr(config, "ESM_LOCAL_MAX_TOKENS", 32),
                )
            )
            global_length = min(wt_lengths[i], mut_lengths[i])
            if any(
                position >= global_length
                for position in context["mutation_positions"]
            ):
                mutation_embs[i] = pool_packed_mutation_esm_features(
                    local_esm[-1]
                )

        for i, fname in enumerate(fnames):
            samples[i]["wt_esm_embedding"] = wt_embs[i]
            samples[i]["mut_esm_embedding"] = mut_embs[i]
            samples[i]["mutation_esm_embedding"] = mutation_embs[i]
            for key, value in local_esm[i].items():
                samples[i][key] = value.detach().cpu()
            samples[i][LOCAL_TOKEN_VERSION_KEY] = getattr(
                config,
                "ESM_LOCAL_TOKEN_VERSION",
                1,
            )
            samples[i]["esm_mutation_window_radius"] = getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8)
            torch.save(samples[i], os.path.join(PRECOMPUTED_DIR, fname))

    write_cache_ready_marker(all_files)
    print("Done!")


if __name__ == "__main__":
    main()
