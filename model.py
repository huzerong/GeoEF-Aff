import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from esm.pretrained import load_model_and_alphabet
from typing import List, Dict, Tuple, Optional
from dynamic_modules import RelationEGNN, EdgeConstructor
from esm_local_tokens import (
    build_local_esm_context,
    pack_preselected_esm_tokens,
)
from protein_features import ProteinFeatureExtractor


MAX_ESM_RESIDUES = 1022


def _zero_init_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            break


def _standardized_aa_physchem() -> torch.Tensor:
    """Charge, side-chain volume, hydrophobicity, and aromaticity in AA index order."""
    values = torch.tensor(
        [
            [0.0, 88.6, 1.8, 0.0],    # A
            [0.0, 108.5, 2.5, 0.0],   # C
            [-1.0, 111.1, -3.5, 0.0], # D
            [-1.0, 138.4, -3.5, 0.0], # E
            [0.0, 189.9, 2.8, 1.0],   # F
            [0.0, 60.1, -0.4, 0.0],   # G
            [0.1, 153.2, -3.2, 1.0],  # H
            [0.0, 166.7, 4.5, 0.0],   # I
            [1.0, 168.6, -3.9, 0.0],  # K
            [0.0, 166.7, 3.8, 0.0],   # L
            [0.0, 162.9, 1.9, 0.0],   # M
            [0.0, 114.1, -3.5, 0.0],  # N
            [0.0, 112.7, -1.6, 0.0],  # P
            [0.0, 143.8, -3.5, 0.0],  # Q
            [1.0, 173.4, -4.5, 0.0],  # R
            [0.0, 89.0, -0.8, 0.0],   # S
            [0.0, 116.1, -0.7, 0.0],  # T
            [0.0, 140.0, 4.2, 0.0],   # V
            [0.0, 227.8, -0.9, 1.0],  # W
            [0.0, 193.6, -1.3, 1.0],  # Y
        ],
        dtype=torch.float32,
    )
    values = (values - values.mean(dim=0, keepdim=True)) / values.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    ).clamp_min(1e-6)
    return torch.cat([values, torch.zeros(1, 4, dtype=torch.float32)], dim=0)


class CrossAttentionLayer(nn.Module): 
    """交叉注意力层，实现两个模态之间的交互关注"""
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 使批次维度在前
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        query_padding_mask=None,
    ):
        # 交叉注意力计算
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # 残差连接 + 层归一化
        query = self.norm(query + self.dropout(attn_output))
        if query_padding_mask is not None:
            query = query.masked_fill(
                query_padding_mask.unsqueeze(-1),
                0.0,
            )
        # 前馈网络
        ffn_output = self.ffn(query)
        query = self.norm(query + self.dropout(ffn_output))
        if query_padding_mask is not None:
            query = query.masked_fill(
                query_padding_mask.unsqueeze(-1),
                0.0,
            )
        return query


class InteractiveAttention(nn.Module):
    """Token-level bidirectional cross-attention between sequence and structure tokens."""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.seq_norm = nn.LayerNorm(hidden_dim)
        self.struct_norm = nn.LayerNorm(hidden_dim)

        # 双向交叉注意力层
        self.seq2struct_attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)
        self.struct2seq_attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)
        
        # 融合输出层
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    @staticmethod
    def _masked_mean(tokens, padding_mask):
        if padding_mask is None:
            return tokens.mean(dim=1)
        valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
        return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        seq_tokens,
        struct_tokens,
        seq_padding_mask=None,
        struct_padding_mask=None,
    ):
        """
        Args:
            seq_tokens: sequence tokens (batch_size, seq_len, hidden_dim)
            struct_tokens: structure tokens (batch_size, struct_len, hidden_dim)
        Returns:
            fused_feats: 融合后的特征 (batch_size, hidden_dim)
        """
        if seq_tokens.dim() == 2:
            seq_tokens = seq_tokens.unsqueeze(1)
        if struct_tokens.dim() == 2:
            struct_tokens = struct_tokens.unsqueeze(1)

        seq_tokens = self.seq_norm(seq_tokens)
        struct_tokens = self.struct_norm(struct_tokens)
        if seq_padding_mask is not None:
            seq_tokens = seq_tokens.masked_fill(
                seq_padding_mask.unsqueeze(-1),
                0.0,
            )
        if struct_padding_mask is not None:
            struct_tokens = struct_tokens.masked_fill(
                struct_padding_mask.unsqueeze(-1),
                0.0,
            )

        # 序列特征关注结构特征
        seq_attn = self.seq2struct_attn(
            query=seq_tokens,
            key=struct_tokens,
            value=struct_tokens,
            key_padding_mask=struct_padding_mask,
            query_padding_mask=seq_padding_mask,
        )
        
        # 结构特征关注序列特征
        struct_attn = self.struct2seq_attn(
            query=struct_tokens,
            key=seq_tokens,
            value=seq_tokens,
            key_padding_mask=seq_padding_mask,
            query_padding_mask=struct_padding_mask,
        )

        seq_pooled = self._masked_mean(seq_attn, seq_padding_mask)
        struct_pooled = self._masked_mean(struct_attn, struct_padding_mask)

        # 特征融合
        fused = torch.cat([seq_pooled, struct_pooled], dim=1)
        fused_feats = self.norm(self.output_proj(fused))
        
        return fused_feats


class ESM_RAAD_FoldX_DDAffinity(nn.Module):

    # 类变量，用于DataParallel时的进程信息
    _rank = 0
    _world_size = 1

    @staticmethod
    def set_parallel_info(rank: int, world_size: int):
        ESM_RAAD_FoldX_DDAffinity._rank = rank
        ESM_RAAD_FoldX_DDAffinity._world_size = world_size

    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        hidden_dim: int = 256,
        raad_hidden_dim: int = 128,
        raad_layers: int = 4,
        dropout: float = 0.1,
        edge_types: int = 8,
        rball_radius: float = 10.0,
        knn_k: int = 10,
        use_atom_features: bool = True,
        use_precomputed_esm: bool = False,
        local_radius: float = 10.0,
        esm_mutation_window_radius: int = 8,
        esm_local_max_tokens: int = 32,
        struct_local_max_residues: int = 32,
        coords_agg: str = "mean",
    ):
        super().__init__()

        self.use_precomputed_esm = use_precomputed_esm
        self.local_radius = local_radius
        self.esm_mutation_window_radius = esm_mutation_window_radius
        self.esm_local_max_tokens = int(esm_local_max_tokens)
        self.struct_local_max_residues = int(struct_local_max_residues)
        if self.esm_local_max_tokens < 1:
            raise ValueError("esm_local_max_tokens must be positive.")
        if self.struct_local_max_residues < 1:
            raise ValueError("struct_local_max_residues must be positive.")
        if coords_agg not in {"mean", "sum"}:
            raise ValueError(f"Unsupported EGNN coordinate aggregation: {coords_agg!r}")

        if use_precomputed_esm:
            # 不加载 ESM 模型，根据模型名推断嵌入维度
            esm_dim_map = {
                "esm2_t6_8M_UR50D": 320,
                "esm2_t12_35M_UR50D": 480,
                "esm2_t30_150M_UR50D": 640,
                "esm2_t33_650M_UR50D": 1280,
                "esm2_t36_3B_UR50D": 2560,
            }
            self.esm_seq_dim = esm_dim_map.get(esm_model_name, 1280)
            self.esm_model = None
            self.esm_alphabet = None
        else:
            self.esm_model, self.esm_alphabet = load_model_and_alphabet(
                esm_model_name
            )
            self.esm_seq_dim = self.esm_model.embed_dim
            for param in self.esm_model.parameters():
                param.requires_grad = False

        self.protein_feature_extractor = ProteinFeatureExtractor(use_atom_features=use_atom_features)
        self.edge_constructor = EdgeConstructor(rball_radius=rball_radius, knn_k=knn_k)
        
        self.raad_gnn = RelationEGNN(
            input_nf=128,
            hidden_nf=raad_hidden_dim,
            output_nf=raad_hidden_dim,
            n_layers=raad_layers,
            edge_type=edge_types,
            dropout=dropout,
            coords_agg=coords_agg,
            attention=True,
            residual=True,
        )

        mutation_aa_dim = 32
        self.mutation_aa_embedding = nn.Embedding(21, mutation_aa_dim)
        self.mutation_type_encoder = nn.Sequential(
            nn.LayerNorm(mutation_aa_dim * 3 + 4),
            nn.Linear(mutation_aa_dim * 3 + 4, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
        )
        self.mutation_type_gate = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.Sigmoid(),
        )
        self.register_buffer(
            "aa_physchem_properties",
            _standardized_aa_physchem(),
            persistent=True,
        )
        _zero_init_last_linear(self.mutation_type_encoder)

        self.seq_proj = nn.Linear(self.esm_seq_dim * 2, hidden_dim)
        self.struct_proj = nn.Linear(raad_hidden_dim, hidden_dim)
        self.seq_local_proj = nn.Sequential(
            nn.LayerNorm(self.esm_seq_dim * 3),
            nn.Linear(self.esm_seq_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.struct_local_proj = nn.Sequential(
            nn.LayerNorm(raad_hidden_dim),
            nn.Linear(raad_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.seq_mutation_flag_embedding = nn.Embedding(2, hidden_dim)

        self.foldx_abs_proj = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.foldx_delta_proj = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.foldx_delta_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        
        # 使用交互注意力替换原有多模态注意力
        self.interactive_attention = InteractiveAttention(
            hidden_dim=hidden_dim,
            num_heads=hidden_dim // 32,  # 每个头32维
            dropout=dropout
        )

        # 调整输出头输入维度
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _get_esm_features(self, antibody_seqs, antigen_seqs, device=None):
        batch_converter = self.esm_alphabet.get_batch_converter()
        complex_seqs = [
            (ab + ag)[:MAX_ESM_RESIDUES]
            for ab, ag in zip(antibody_seqs, antigen_seqs)
        ]

        data = [("complex", seq) for seq in complex_seqs]
        _, _, batch_tokens = batch_converter(data)
        
        if device is None:
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.esm_model(
                batch_tokens,
                repr_layers=[self.esm_model.num_layers],
                return_contacts=False,
            )

        token_representations = results["representations"][self.esm_model.num_layers]
        seq_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)
        token_representations = token_representations[:, 1:-1]

        global_seq_feats = []
        for i, tokens_len in enumerate(seq_lens):
            global_seq_feats.append(token_representations[i, : tokens_len - 2].mean(0))
        seq_feats = torch.stack(global_seq_feats, dim=0)
        return seq_feats

    @staticmethod
    def _decode_sequence_batch(seqs):
        if isinstance(seqs, torch.Tensor):
            decoded = []
            for row in seqs.detach().cpu().tolist():
                decoded.append("".join(chr(int(c)) for c in row if int(c) > 0))
            return decoded
        return list(seqs)

    @staticmethod
    def _find_mutation_positions(wt_seq: str, mut_seq: str, max_len: int) -> List[int]:
        limit = min(len(wt_seq), len(mut_seq), max_len)
        return [idx for idx in range(limit) if wt_seq[idx] != mut_seq[idx]]

    def _get_esm_features_optimized(self, antibody_seqs, antigen_seqs, device=None, return_tokens=False):
        batch_converter = self.esm_alphabet.get_batch_converter()
        complex_seqs = [
            (ab + ag)[:MAX_ESM_RESIDUES]
            for ab, ag in zip(antibody_seqs, antigen_seqs)
        ]

        data = [("complex", seq) for seq in complex_seqs]
        _, _, batch_tokens = batch_converter(data)
        
        if device is None:
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = self.esm_model(
                batch_tokens,
                repr_layers=[self.esm_model.num_layers],
                return_contacts=False,
            )

        token_representations = results["representations"][self.esm_model.num_layers]
        seq_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)
        token_representations = token_representations[:, 1:-1]

        global_seq_feats = []
        actual_lengths = []
        for i, tokens_len in enumerate(seq_lens):
            actual_len = int(tokens_len.item()) - 2
            actual_lengths.append(actual_len)
            global_seq_feats.append(token_representations[i, :actual_len].mean(0))
        seq_feats = torch.stack(global_seq_feats, dim=0)

        if return_tokens:
            return seq_feats, token_representations, actual_lengths

        del results
        del batch_tokens
        del token_representations

        return seq_feats

    def _pool_mutation_esm_features(
        self,
        wt_tokens: torch.Tensor,
        mut_tokens: torch.Tensor,
        wt_lengths: List[int],
        mut_lengths: List[int],
        wt_complex_seqs: List[str],
        mut_complex_seqs: List[str],
    ) -> torch.Tensor:
        local_features = []
        radius = int(self.esm_mutation_window_radius)

        for batch_idx, (wt_seq, mut_seq) in enumerate(zip(wt_complex_seqs, mut_complex_seqs)):
            max_len = min(wt_lengths[batch_idx], mut_lengths[batch_idx])
            mut_positions = self._find_mutation_positions(wt_seq, mut_seq, max_len)
            if not mut_positions:
                local_features.append(wt_tokens.new_zeros(self.esm_seq_dim * 4))
                continue

            pos_tensor = torch.tensor(mut_positions, dtype=torch.long, device=wt_tokens.device)
            wt_site = wt_tokens[batch_idx, pos_tensor].mean(dim=0)
            mut_site = mut_tokens[batch_idx, pos_tensor].mean(dim=0)

            window_mask = torch.zeros(max_len, dtype=torch.bool, device=wt_tokens.device)
            for pos in mut_positions:
                start = max(0, pos - radius)
                end = min(max_len, pos + radius + 1)
                window_mask[start:end] = True
            window_idx = torch.nonzero(window_mask, as_tuple=True)[0]
            wt_window = wt_tokens[batch_idx, window_idx].mean(dim=0)
            mut_window = mut_tokens[batch_idx, window_idx].mean(dim=0)

            local_features.append(
                torch.cat(
                    [
                        wt_site,
                        mut_site,
                        mut_site - wt_site,
                        mut_window - wt_window,
                    ],
                    dim=0,
                )
            )

        return torch.stack(local_features, dim=0)

    def _pack_runtime_local_esm(
        self,
        wt_tokens,
        mut_tokens,
        wt_lengths,
        mut_lengths,
        wt_sequences,
        mutant_sequences,
    ):
        contexts = [
            build_local_esm_context(
                wt_sequence=wt_sequence,
                mutant_sequence=mutant_sequence,
                radius=self.esm_mutation_window_radius,
                max_tokens=self.esm_local_max_tokens,
                max_context_length=MAX_ESM_RESIDUES,
            )
            for wt_sequence, mutant_sequence in zip(
                wt_sequences,
                mutant_sequences,
            )
        ]
        local_wt_tokens = [None] * len(contexts)
        local_mut_tokens = [None] * len(contexts)
        crop_indices = []
        crop_wt_sequences = []
        crop_mut_sequences = []
        for batch_idx, context in enumerate(contexts):
            if context["context_start"] == 0:
                local_wt_tokens[batch_idx] = wt_tokens[
                    batch_idx,
                    : wt_lengths[batch_idx],
                ]
                local_mut_tokens[batch_idx] = mut_tokens[
                    batch_idx,
                    : mut_lengths[batch_idx],
                ]
            else:
                crop_indices.append(batch_idx)
                crop_wt_sequences.append(context["wt_context_sequence"])
                crop_mut_sequences.append(context["mutant_context_sequence"])

        if crop_indices:
            empty_partners = [""] * len(crop_indices)
            _, crop_wt_tokens, crop_wt_lengths = (
                self._get_esm_features_optimized(
                    crop_wt_sequences,
                    empty_partners,
                    device=wt_tokens.device,
                    return_tokens=True,
                )
            )
            _, crop_mut_tokens, crop_mut_lengths = (
                self._get_esm_features_optimized(
                    crop_mut_sequences,
                    empty_partners,
                    device=mut_tokens.device,
                    return_tokens=True,
                )
            )
            for crop_idx, batch_idx in enumerate(crop_indices):
                local_wt_tokens[batch_idx] = crop_wt_tokens[
                    crop_idx,
                    : crop_wt_lengths[crop_idx],
                ]
                local_mut_tokens[batch_idx] = crop_mut_tokens[
                    crop_idx,
                    : crop_mut_lengths[crop_idx],
                ]

        packed = []
        for batch_idx, context in enumerate(contexts):
            packed.append(
                pack_preselected_esm_tokens(
                    wt_context_tokens=local_wt_tokens[batch_idx],
                    mutant_context_tokens=local_mut_tokens[batch_idx],
                    context_token_indices=context["context_token_indices"],
                    selected_positions=context["selected_positions"],
                    mutation_positions=context["mutation_positions"],
                    max_tokens=self.esm_local_max_tokens,
                )
            )
        return (
            torch.stack(
                [item["wt_esm_window_tokens"] for item in packed],
                dim=0,
            ),
            torch.stack(
                [item["mut_esm_window_tokens"] for item in packed],
                dim=0,
            ),
            torch.stack(
                [item["esm_window_padding_mask"] for item in packed],
                dim=0,
            ),
            torch.stack(
                [item["esm_window_mutation_mask"] for item in packed],
                dim=0,
            ),
        )

    def _build_foldx_features(
        self,
        foldx_energies: torch.Tensor,
        foldx_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if foldx_features is not None:
            foldx_features = foldx_features.to(foldx_energies.device, dtype=foldx_energies.dtype)
            if foldx_features.dim() == 1:
                foldx_features = foldx_features.unsqueeze(0)
            if foldx_features.shape[-1] == 3:
                return foldx_features
            raise ValueError(
                f"Expected foldx_features with shape [batch, 3] for "
                f"[wt_energy, mut_energy, mut_energy - wt_energy], got {tuple(foldx_features.shape)}"
            )

        wt_energy = foldx_energies.view(-1, 1)
        zeros = torch.zeros_like(wt_energy)
        return torch.cat([wt_energy, wt_energy, zeros], dim=1)

    def forward(
        self,
        antibody_seqs: List[str],
        antigen_seqs: List[str],
        mutant_antibody_seqs: List[str],
        mutant_antigen_seqs: List[str],
        foldx_energies: torch.Tensor,
        structure_data: Optional[Dict[str, torch.Tensor]] = None,
        wt_esm_embedding: Optional[torch.Tensor] = None,
        mut_esm_embedding: Optional[torch.Tensor] = None,
        mutation_esm_embedding: Optional[torch.Tensor] = None,
        wt_esm_window_tokens: Optional[torch.Tensor] = None,
        mut_esm_window_tokens: Optional[torch.Tensor] = None,
        esm_window_padding_mask: Optional[torch.Tensor] = None,
        esm_window_mutation_mask: Optional[torch.Tensor] = None,
        foldx_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use foldx_energies size as batch_size since DataParallel splits tensors
        batch_size = foldx_energies.shape[0]

        # Handle empty batch (can happen with DataParallel on last batch)
        if batch_size == 0:
            return torch.zeros(0, device=foldx_energies.device)

        device = foldx_energies.device
        has_wt_precomputed = wt_esm_embedding is not None
        has_mut_precomputed = mut_esm_embedding is not None
        if has_wt_precomputed != has_mut_precomputed:
            raise ValueError(
                "WT and mutant global ESM embeddings must be provided together."
            )
        if not has_wt_precomputed and self.esm_model is None:
            raise ValueError(
                "This model was created for precomputed ESM input, but no "
                "global ESM embeddings were provided."
            )

        # 如果有预计算的 ESM 嵌入，直接使用，跳过 ESM forward
        if wt_esm_embedding is not None and mut_esm_embedding is not None:
            wt_seq_feats = wt_esm_embedding.to(device)
            mut_seq_feats = mut_esm_embedding.to(device)
            local_inputs = (
                wt_esm_window_tokens,
                mut_esm_window_tokens,
                esm_window_padding_mask,
                esm_window_mutation_mask,
            )
            if any(value is None for value in local_inputs):
                raise ValueError(
                    "Precomputed ESM mode requires WT/Mut local tokens, "
                    "padding mask, and mutation mask."
                )
            wt_local_tokens = wt_esm_window_tokens.to(device)
            mut_local_tokens = mut_esm_window_tokens.to(device)
            seq_local_padding_mask = esm_window_padding_mask.to(
                device=device,
                dtype=torch.bool,
            )
            seq_local_mutation_mask = esm_window_mutation_mask.to(
                device=device,
                dtype=torch.bool,
            )
        else:
            # 回退到运行时 ESM 推理
            antibody_seqs = self._decode_sequence_batch(antibody_seqs)
            antigen_seqs = self._decode_sequence_batch(antigen_seqs)
            mutant_antibody_seqs = self._decode_sequence_batch(mutant_antibody_seqs)
            mutant_antigen_seqs = self._decode_sequence_batch(mutant_antigen_seqs)

            wt_seq_feats, wt_tokens, wt_lengths = self._get_esm_features_optimized(
                antibody_seqs, antigen_seqs, device=device, return_tokens=True
            )
            mut_seq_feats, mut_tokens, mut_lengths = self._get_esm_features_optimized(
                mutant_antibody_seqs, mutant_antigen_seqs, device=device, return_tokens=True
            )
            wt_complex_sequences = [
                ab + ag for ab, ag in zip(antibody_seqs, antigen_seqs)
            ]
            mutant_complex_sequences = [
                ab + ag
                for ab, ag in zip(
                    mutant_antibody_seqs,
                    mutant_antigen_seqs,
                )
            ]
            (
                wt_local_tokens,
                mut_local_tokens,
                seq_local_padding_mask,
                seq_local_mutation_mask,
            ) = self._pack_runtime_local_esm(
                wt_tokens=wt_tokens,
                mut_tokens=mut_tokens,
                wt_lengths=wt_lengths,
                mut_lengths=mut_lengths,
                wt_sequences=wt_complex_sequences,
                mutant_sequences=mutant_complex_sequences,
            )

            if isinstance(wt_seq_feats, tuple):
                wt_seq_feats = wt_seq_feats[0]
            if isinstance(mut_seq_feats, tuple):
                mut_seq_feats = mut_seq_feats[0]

        expected_global_shape = (batch_size, self.esm_seq_dim)
        expected_local_shape = (
            batch_size,
            self.esm_local_max_tokens,
            self.esm_seq_dim,
        )
        expected_mask_shape = (
            batch_size,
            self.esm_local_max_tokens,
        )
        if tuple(wt_seq_feats.shape) != expected_global_shape:
            raise ValueError(
                "WT global ESM embedding shape mismatch: "
                f"{tuple(wt_seq_feats.shape)} != {expected_global_shape}."
            )
        if tuple(mut_seq_feats.shape) != expected_global_shape:
            raise ValueError(
                "Mutant global ESM embedding shape mismatch: "
                f"{tuple(mut_seq_feats.shape)} != {expected_global_shape}."
            )
        if tuple(wt_local_tokens.shape) != expected_local_shape:
            raise ValueError(
                "WT local ESM token shape mismatch: "
                f"{tuple(wt_local_tokens.shape)} != {expected_local_shape}."
            )
        if tuple(mut_local_tokens.shape) != expected_local_shape:
            raise ValueError(
                "Mutant local ESM token shape mismatch: "
                f"{tuple(mut_local_tokens.shape)} != {expected_local_shape}."
            )
        if tuple(seq_local_padding_mask.shape) != expected_mask_shape:
            raise ValueError(
                "Local ESM padding-mask shape mismatch: "
                f"{tuple(seq_local_padding_mask.shape)} != "
                f"{expected_mask_shape}."
            )
        if tuple(seq_local_mutation_mask.shape) != expected_mask_shape:
            raise ValueError(
                "Local ESM mutation-mask shape mismatch: "
                f"{tuple(seq_local_mutation_mask.shape)} != "
                f"{expected_mask_shape}."
            )
        seq_feats = torch.cat([wt_seq_feats, mut_seq_feats], dim=1)

        if structure_data is not None:
            coords = structure_data['coords']
            aa_types = structure_data['aa_types']
            atom_types = structure_data['atom_types']
            segment_ids = structure_data['segment_ids']
            seq_idx = structure_data['seq_idx']
            batch_ids = structure_data.get('batch_ids', torch.zeros_like(segment_ids))
            chain_numeric_ids = structure_data.get("chain_numeric_ids")
            mutation_mask = structure_data.get('mutation_mask')
            mutation_ca_mask = structure_data.get('mutation_ca_mask')
            ca_mask = structure_data.get('ca_mask')
            residue_uid = structure_data.get('residue_uid')
            wt_aa_types = structure_data.get('wt_aa_types')
            mutant_aa_types = structure_data.get('mutant_aa_types')
            if (
                ca_mask is None
                or residue_uid is None
                or chain_numeric_ids is None
            ):
                raise ValueError(
                    "Local structure tokens require ca_mask, residue_uid, "
                    "and chain_numeric_ids."
                )

            node_features = self.protein_feature_extractor(
                coords, aa_types, atom_types, segment_ids
            )

            if mutation_mask is not None:
                if wt_aa_types is None or mutant_aa_types is None:
                    raise ValueError(
                        "Mutation-aware structure modeling requires wt_aa_types and mutant_aa_types."
                    )
                wt_aa_types = wt_aa_types.to(device=device, dtype=torch.long)
                mutant_aa_types = mutant_aa_types.to(device=device, dtype=torch.long)
                if wt_aa_types.shape != aa_types.shape or mutant_aa_types.shape != aa_types.shape:
                    raise ValueError("WT/Mut amino-acid type tensors must align with structure atoms.")

                wt_aa_emb = self.mutation_aa_embedding(wt_aa_types)
                mut_aa_emb = self.mutation_aa_embedding(mutant_aa_types)
                physchem_delta = (
                    self.aa_physchem_properties[mutant_aa_types]
                    - self.aa_physchem_properties[wt_aa_types]
                ).to(dtype=node_features.dtype)
                mutation_input = torch.cat(
                    [
                        wt_aa_emb,
                        mut_aa_emb,
                        mut_aa_emb - wt_aa_emb,
                        physchem_delta,
                    ],
                    dim=-1,
                ).to(dtype=node_features.dtype)
                mutation_update = self.mutation_type_encoder(mutation_input)
                mutation_gate = self.mutation_type_gate(
                    torch.cat([node_features, mutation_update], dim=-1)
                )
                mutation_node_mask = mutation_mask.to(
                    device=device,
                    dtype=node_features.dtype,
                ).unsqueeze(-1)
                node_features = (
                    node_features
                    + mutation_node_mask * mutation_gate * mutation_update
                )

            edge_list = self.edge_constructor.build_edges(
                coords,
                segment_ids,
                batch_ids,
                seq_idx,
                chain_ids=chain_numeric_ids,
            )

            struct_feats, updated_coords = self.raad_gnn(
                node_features, coords, edge_list, segment_ids
            )
            
            (
                struct_feats_aggregated,
                struct_local_feats,
                struct_local_padding_mask,
            ) = self._build_residue_structure_tokens(
                struct_feats,
                coords,
                batch_ids,
                residue_uid,
                ca_mask,
                batch_size,
                mutation_mask=mutation_mask,
                mutation_ca_mask=mutation_ca_mask,
            )
        else:
            struct_feats_aggregated = torch.zeros(
                batch_size, self.raad_gnn.output_nf
            ).to(seq_feats.device)
            struct_local_feats = torch.zeros(
                batch_size,
                self.struct_local_max_residues,
                self.raad_gnn.output_nf,
            ).to(seq_feats.device)
            struct_local_padding_mask = torch.ones(
                batch_size,
                self.struct_local_max_residues,
                dtype=torch.bool,
                device=seq_feats.device,
            )

        seq_global_proj = self.seq_proj(seq_feats)
        seq_local_input = torch.cat(
            [
                wt_local_tokens,
                mut_local_tokens,
                mut_local_tokens - wt_local_tokens,
            ],
            dim=-1,
        )
        seq_local_proj = self.seq_local_proj(seq_local_input)
        mutation_flag_embedding = self.seq_mutation_flag_embedding(
            seq_local_mutation_mask.long()
        ).to(seq_local_proj.dtype)
        seq_local_proj = (
            seq_local_proj
            + mutation_flag_embedding
        )
        seq_local_proj = seq_local_proj.masked_fill(
            seq_local_padding_mask.unsqueeze(-1),
            0.0,
        )
        struct_global_proj = self.struct_proj(struct_feats_aggregated)
        struct_local_proj = self.struct_local_proj(struct_local_feats)
        struct_local_proj = struct_local_proj.masked_fill(
            struct_local_padding_mask.unsqueeze(-1),
            0.0,
        )
        seq_tokens = torch.cat(
            [seq_global_proj.unsqueeze(1), seq_local_proj],
            dim=1,
        )
        struct_tokens = torch.cat(
            [struct_global_proj.unsqueeze(1), struct_local_proj],
            dim=1,
        )
        global_padding = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=device,
        )
        seq_padding_mask = torch.cat(
            [global_padding, seq_local_padding_mask],
            dim=1,
        )
        struct_padding_mask = torch.cat(
            [global_padding, struct_local_padding_mask],
            dim=1,
        )

        fused_feats = self.interactive_attention(
            seq_tokens,
            struct_tokens,
            seq_padding_mask=seq_padding_mask,
            struct_padding_mask=struct_padding_mask,
        )

        foldx_feature_tensor = self._build_foldx_features(foldx_energies, foldx_features)
        foldx_abs_emb = self.foldx_abs_proj(foldx_feature_tensor[:, :2])
        foldx_delta_emb = self.foldx_delta_proj(foldx_feature_tensor[:, 2:3])
        foldx_gate = self.foldx_delta_gate(torch.cat([fused_feats, foldx_delta_emb], dim=1))
        foldx_emb = foldx_abs_emb + foldx_gate * foldx_delta_emb

        final_feats = torch.cat([fused_feats, foldx_emb], dim=1)

        pred_delta_g = self.affinity_head(final_feats).squeeze(-1)

        return pred_delta_g
    
    def _build_residue_structure_tokens(
        self,
        struct_feats: torch.Tensor,
        coords: torch.Tensor,
        batch_ids: torch.Tensor,
        residue_uid: torch.Tensor,
        ca_mask: torch.Tensor,
        batch_size: int,
        mutation_mask: Optional[torch.Tensor] = None,
        mutation_ca_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_dim = int(struct_feats.shape[1])
        device = struct_feats.device
        atom_count = int(struct_feats.shape[0])
        if atom_count == 0:
            raise ValueError("Cannot build residue tokens from an empty structure.")

        batch_ids = batch_ids.to(device=device, dtype=torch.long)
        residue_uid = residue_uid.to(device=device, dtype=torch.long)
        ca_mask = ca_mask.to(device=device, dtype=torch.bool)
        if residue_uid.shape[0] != atom_count or ca_mask.shape[0] != atom_count:
            raise ValueError("residue_uid and ca_mask must align with structure atoms.")
        if mutation_mask is None:
            mutation_mask = torch.zeros(atom_count, dtype=torch.bool, device=device)
        else:
            mutation_mask = mutation_mask.to(device=device, dtype=torch.bool)
            if mutation_mask.shape[0] != atom_count:
                raise ValueError("mutation_mask must align with structure atoms.")
        if mutation_ca_mask is None:
            mutation_ca_mask = mutation_mask & ca_mask
        else:
            mutation_ca_mask = mutation_ca_mask.to(
                device=device,
                dtype=torch.bool,
            )
            if mutation_ca_mask.shape[0] != atom_count:
                raise ValueError(
                    "mutation_ca_mask must align with structure atoms."
                )

        residue_count = int(residue_uid.max().item()) + 1
        atom_weights = struct_feats.new_ones((atom_count, 1))
        residue_atom_counts = struct_feats.new_zeros((residue_count, 1))
        residue_atom_counts.index_add_(0, residue_uid, atom_weights)

        residue_features = struct_feats.new_zeros(
            (residue_count, hidden_dim)
        )
        residue_features.index_add_(0, residue_uid, struct_feats)
        residue_features = residue_features / residue_atom_counts.clamp_min(1.0)

        residue_coord_sums = coords.new_zeros((residue_count, 3))
        residue_coord_sums.index_add_(0, residue_uid, coords)
        residue_coords = residue_coord_sums / residue_atom_counts.clamp_min(1.0)

        ca_weights = ca_mask.to(coords.dtype).unsqueeze(-1)
        residue_ca_counts = coords.new_zeros((residue_count, 1))
        residue_ca_counts.index_add_(0, residue_uid, ca_weights)
        residue_ca_sums = coords.new_zeros((residue_count, 3))
        residue_ca_sums.index_add_(0, residue_uid, coords * ca_weights)
        ca_available = residue_ca_counts.squeeze(-1) > 0
        residue_ca_coords = (
            residue_ca_sums
            / residue_ca_counts.clamp_min(1.0)
        )
        residue_coords = torch.where(
            ca_available.unsqueeze(-1),
            residue_ca_coords,
            residue_coords,
        )

        residue_batch_ids = torch.zeros(
            residue_count,
            dtype=torch.long,
            device=device,
        )
        residue_batch_ids.scatter_(0, residue_uid, batch_ids)
        residue_mutation_counts = struct_feats.new_zeros((residue_count, 1))
        residue_mutation_counts.index_add_(
            0,
            residue_uid,
            mutation_mask.to(struct_feats.dtype).unsqueeze(-1),
        )
        residue_is_mutation = residue_mutation_counts.squeeze(-1) > 0

        global_features = []
        local_features = struct_feats.new_zeros(
            (
                batch_size,
                self.struct_local_max_residues,
                hidden_dim,
            )
        )
        local_padding_mask = torch.ones(
            batch_size,
            self.struct_local_max_residues,
            dtype=torch.bool,
            device=device,
        )

        for batch_index in range(batch_size):
            sample_residue_indices = torch.nonzero(
                residue_batch_ids == batch_index,
                as_tuple=True,
            )[0]
            if sample_residue_indices.numel() == 0:
                global_features.append(
                    struct_feats.new_zeros(hidden_dim)
                )
                continue

            sample_residue_features = residue_features[sample_residue_indices]
            global_features.append(sample_residue_features.mean(dim=0))
            sample_atom_mask = batch_ids == batch_index
            centers = coords[sample_atom_mask & mutation_ca_mask]
            sample_mutation_mask = residue_is_mutation[sample_residue_indices]
            mutation_residue_indices = sample_residue_indices[
                sample_mutation_mask
            ]
            if centers.numel() == 0 and mutation_residue_indices.numel() > 0:
                centers = residue_coords[mutation_residue_indices]
            if centers.numel() == 0:
                raise ValueError(
                    f"Sample {batch_index} has no mutation CA center."
                )
            if (
                mutation_residue_indices.numel()
                > self.struct_local_max_residues
            ):
                raise ValueError(
                    "Mutation residues exceed STRUCT_LOCAL_MAX_RESIDUES: "
                    f"{mutation_residue_indices.numel()} > "
                    f"{self.struct_local_max_residues}."
                )

            distances = torch.cdist(
                residue_coords[sample_residue_indices],
                centers,
            ).min(dim=1).values
            mutation_local_positions = torch.nonzero(
                sample_mutation_mask,
                as_tuple=True,
            )[0]
            non_mutation_local_positions = torch.nonzero(
                ~sample_mutation_mask,
                as_tuple=True,
            )[0]
            ordered_non_mutation = non_mutation_local_positions[
                torch.argsort(distances[non_mutation_local_positions])
            ]
            selected_local_positions = torch.cat(
                [mutation_local_positions, ordered_non_mutation],
                dim=0,
            )[: self.struct_local_max_residues]
            selected_residue_indices = sample_residue_indices[
                selected_local_positions
            ]
            selected_count = int(selected_residue_indices.numel())
            local_features[batch_index, :selected_count] = residue_features[
                selected_residue_indices
            ]
            local_padding_mask[batch_index, :selected_count] = False

        return (
            torch.stack(global_features, dim=0),
            local_features,
            local_padding_mask,
        )


class ESM_FoldX_DDAffinity(ESM_RAAD_FoldX_DDAffinity):
    
    def __init__(self, **kwargs):
        raad_params = ['raad_hidden_dim', 'raad_layers', 'edge_types', 
                       'rball_radius', 'knn_k', 'use_atom_features']
        for param in raad_params:
            kwargs.pop(param, None)
        
        super().__init__(** kwargs)
    
    def forward(self, antibody_seqs, antigen_seqs, mutant_antibody_seqs, mutant_antigen_seqs, foldx_energies,
                wt_esm_embedding=None, mut_esm_embedding=None, mutation_esm_embedding=None,
                wt_esm_window_tokens=None, mut_esm_window_tokens=None,
                esm_window_padding_mask=None, esm_window_mutation_mask=None,
                foldx_features=None):
        return super().forward(
            antibody_seqs,
            antigen_seqs,
            mutant_antibody_seqs,
            mutant_antigen_seqs,
            foldx_energies,
            structure_data=None,
            wt_esm_embedding=wt_esm_embedding,
            mut_esm_embedding=mut_esm_embedding,
            mutation_esm_embedding=mutation_esm_embedding,
            wt_esm_window_tokens=wt_esm_window_tokens,
            mut_esm_window_tokens=mut_esm_window_tokens,
            esm_window_padding_mask=esm_window_padding_mask,
            esm_window_mutation_mask=esm_window_mutation_mask,
            foldx_features=foldx_features,
        )
