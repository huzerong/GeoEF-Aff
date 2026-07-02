"""
预计算ESM-2嵌入并写入已有.pt文件
训练时直接加载跳过forwardpass
"""
import os
import torch
from esm.pretrained import load_model_and_alphabet
from tqdm import tqdm

import config

PRECOMPUTED_DIR = config.PRECOMPUTED_DIR
BATCH_SIZE = 8  
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


def main():
    all_files = sorted(
        [f for f in os.listdir(PRECOMPUTED_DIR) if f.endswith(".pt")],
        key=lambda x: int(x.replace(".pt", "")),
    )
    print(f"Total precomputed samples: {len(all_files)}")

    #检查嵌入
    todo = []
    for fname in all_files:
        path = os.path.join(PRECOMPUTED_DIR, fname)
        sample = torch.load(path, weights_only=False)
        foldx_features = sample.get("foldx_features")
        if not isinstance(foldx_features, torch.Tensor) or int(foldx_features.numel()) != getattr(config, "FOLDX_FEATURE_DIM", 3):
            raise ValueError(f"{path} is missing 3-dimensional foldx_features.")
        if sample.get("foldx_feature_mode") != getattr(config, "FOLDX_FEATURE_MODE", "wt_mut_delta"):
            raise ValueError(f"{path} has stale foldx_feature_mode={sample.get('foldx_feature_mode')!r}.")
        required_keys = {"wt_esm_embedding", "mut_esm_embedding", "mutation_esm_embedding"}
        if (
            not required_keys.issubset(sample.keys())
            or sample.get("esm_mutation_window_radius") != getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8)
        ):
            todo.append((fname, sample))

    print(f"Samples needing ESM embeddings: {len(todo)}")
    if not todo:
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

        for i, fname in enumerate(fnames):
            samples[i]["wt_esm_embedding"] = wt_embs[i]
            samples[i]["mut_esm_embedding"] = mut_embs[i]
            samples[i]["mutation_esm_embedding"] = mutation_embs[i]
            samples[i]["esm_mutation_window_radius"] = getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8)
            torch.save(samples[i], os.path.join(PRECOMPUTED_DIR, fname))

    print("Done!")


if __name__ == "__main__":
    main()
