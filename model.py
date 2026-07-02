import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from esm.pretrained import load_model_and_alphabet
from typing import List, Dict, Tuple, Optional
from dynamic_modules import RelationEGNN, EdgeConstructor
from protein_features import ProteinFeatureExtractor


def _zero_init_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            break


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
    
    def forward(self, query, key, value, attn_mask=None):
        # 交叉注意力计算
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask
        )
        # 残差连接 + 层归一化
        query = self.norm(query + self.dropout(attn_output))
        # 前馈网络
        ffn_output = self.ffn(query)
        return self.norm(query + self.dropout(ffn_output))


class InteractiveAttention(nn.Module):
    """交互注意力模块，实现序列与结构特征的双向交叉关注"""
    def __init__(self, seq_dim, struct_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.seq_dim = seq_dim
        self.struct_dim = struct_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 特征维度对齐
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = nn.Linear(struct_dim, hidden_dim)
        
        # 双向交叉注意力层
        self.seq2struct_attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)
        self.struct2seq_attn = CrossAttentionLayer(hidden_dim, num_heads, dropout)
        
        # 融合输出层
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, seq_feats, struct_feats):
        """
        Args:
            seq_feats: 序列特征 (batch_size, seq_dim)
            struct_feats: 结构特征 (batch_size, struct_dim)
        Returns:
            fused_feats: 融合后的特征 (batch_size, hidden_dim)
        """
        # 扩展维度以适应注意力机制 (batch_size, 1, hidden_dim)
        seq_proj = self.seq_proj(seq_feats).unsqueeze(1)
        struct_proj = self.struct_proj(struct_feats).unsqueeze(1)
        
        # 序列特征关注结构特征
        seq_attn = self.seq2struct_attn(
            query=seq_proj,
            key=struct_proj,
            value=struct_proj
        ).squeeze(1)  # (batch_size, hidden_dim)
        
        # 结构特征关注序列特征
        struct_attn = self.struct2seq_attn(
            query=struct_proj,
            key=seq_proj,
            value=seq_proj
        ).squeeze(1)  # (batch_size, hidden_dim)
        
        # 特征融合
        fused = torch.cat([seq_attn, struct_attn], dim=1)
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
        esm_model_name: str = "esm2_t30_150M_UR50D",
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
    ):
        super().__init__()

        self.use_precomputed_esm = use_precomputed_esm
        self.local_radius = local_radius
        self.esm_mutation_window_radius = esm_mutation_window_radius

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
            attention=True,
            residual=True,
        )

        self.seq_proj = nn.Linear(self.esm_seq_dim * 2, hidden_dim)
        self.struct_proj = nn.Linear(raad_hidden_dim, hidden_dim)
        self.seq_local_proj = nn.Sequential(
            nn.LayerNorm(self.esm_seq_dim * 4),
            nn.Linear(self.esm_seq_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.struct_local_proj = nn.Sequential(
            nn.LayerNorm(raad_hidden_dim * 2),
            nn.Linear(raad_hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.esm_mutation_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        _zero_init_last_linear(self.seq_local_proj)
        _zero_init_last_linear(self.struct_local_proj)

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
            seq_dim=hidden_dim,
            struct_dim=hidden_dim,
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
        complex_seqs = [ab + ag for ab, ag in zip(antibody_seqs, antigen_seqs)]

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
        complex_seqs = [ab + ag for ab, ag in zip(antibody_seqs, antigen_seqs)]

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
        foldx_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use foldx_energies size as batch_size since DataParallel splits tensors
        batch_size = foldx_energies.shape[0]

        # Handle empty batch (can happen with DataParallel on last batch)
        if batch_size == 0:
            return torch.zeros(0, device=foldx_energies.device)

        device = foldx_energies.device

        # 如果有预计算的 ESM 嵌入，直接使用，跳过 ESM forward
        if wt_esm_embedding is not None and mut_esm_embedding is not None:
            wt_seq_feats = wt_esm_embedding.to(device)
            mut_seq_feats = mut_esm_embedding.to(device)
            if mutation_esm_embedding is not None:
                mutation_esm_feats = mutation_esm_embedding.to(device)
            else:
                mutation_esm_feats = torch.zeros(
                    batch_size,
                    self.esm_seq_dim * 4,
                    dtype=wt_seq_feats.dtype,
                    device=device,
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
            mutation_esm_feats = self._pool_mutation_esm_features(
                wt_tokens=wt_tokens,
                mut_tokens=mut_tokens,
                wt_lengths=wt_lengths,
                mut_lengths=mut_lengths,
                wt_complex_seqs=[ab + ag for ab, ag in zip(antibody_seqs, antigen_seqs)],
                mut_complex_seqs=[ab + ag for ab, ag in zip(mutant_antibody_seqs, mutant_antigen_seqs)],
            )

            if isinstance(wt_seq_feats, tuple):
                wt_seq_feats = wt_seq_feats[0]
            if isinstance(mut_seq_feats, tuple):
                mut_seq_feats = mut_seq_feats[0]

        seq_feats = torch.cat([wt_seq_feats, mut_seq_feats], dim=1)

        if structure_data is not None:
            coords = structure_data['coords']
            aa_types = structure_data['aa_types']
            atom_types = structure_data['atom_types']
            segment_ids = structure_data['segment_ids']
            seq_idx = structure_data['seq_idx']
            batch_ids = structure_data.get('batch_ids', torch.zeros_like(segment_ids))
            mutation_mask = structure_data.get('mutation_mask')
            mutation_ca_mask = structure_data.get('mutation_ca_mask')

            node_features = self.protein_feature_extractor(
                coords, aa_types, atom_types, segment_ids
            )

            edge_list = self.edge_constructor.build_edges(
                coords, segment_ids, batch_ids, seq_idx
            )

            struct_feats, updated_coords = self.raad_gnn(
                node_features, coords, edge_list, segment_ids
            )
            
            struct_feats_aggregated, struct_local_feats = self._aggregate_structure_features(
                struct_feats,
                coords,
                batch_ids,
                batch_size,
                mutation_mask=mutation_mask,
                mutation_ca_mask=mutation_ca_mask,
            )
        else:
            struct_feats_aggregated = torch.zeros(
                batch_size, self.raad_gnn.output_nf
            ).to(seq_feats.device)
            struct_local_feats = torch.zeros(
                batch_size, self.raad_gnn.output_nf * 2
            ).to(seq_feats.device)

        seq_global_proj = self.seq_proj(seq_feats)
        seq_mutation_proj = self.seq_local_proj(mutation_esm_feats)
        seq_mutation_gate = self.esm_mutation_gate(
            torch.cat([seq_global_proj, seq_mutation_proj], dim=1)
        )
        seq_feats_proj = seq_global_proj + seq_mutation_gate * seq_mutation_proj
        struct_feats_proj = self.struct_proj(struct_feats_aggregated) + self.struct_local_proj(struct_local_feats)

        # 使用交互注意力融合特征
        fused_feats = self.interactive_attention(seq_feats_proj, struct_feats_proj)

        foldx_feature_tensor = self._build_foldx_features(foldx_energies, foldx_features)
        foldx_abs_emb = self.foldx_abs_proj(foldx_feature_tensor[:, :2])
        foldx_delta_emb = self.foldx_delta_proj(foldx_feature_tensor[:, 2:3])
        foldx_gate = self.foldx_delta_gate(torch.cat([fused_feats, foldx_delta_emb], dim=1))
        foldx_emb = foldx_abs_emb + foldx_gate * foldx_delta_emb

        final_feats = torch.cat([fused_feats, foldx_emb], dim=1)

        pred_delta_g = self.affinity_head(final_feats).squeeze(-1)

        return pred_delta_g
    
    def _aggregate_structure_features(
        self, 
        struct_feats: torch.Tensor, 
        coords: torch.Tensor,
        batch_ids: torch.Tensor, 
        batch_size: int,
        mutation_mask: Optional[torch.Tensor] = None,
        mutation_ca_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aggregated_feats = []
        local_feats = []
        hidden_dim = struct_feats.shape[1]
        device = struct_feats.device

        if mutation_mask is None:
            mutation_mask = torch.zeros(struct_feats.shape[0], dtype=torch.bool, device=device)
        else:
            mutation_mask = mutation_mask.to(device=device, dtype=torch.bool)

        if mutation_ca_mask is None:
            mutation_ca_mask = mutation_mask
        else:
            mutation_ca_mask = mutation_ca_mask.to(device=device, dtype=torch.bool)
        
        for i in range(batch_size):
            batch_mask = batch_ids == i
            if batch_mask.sum() > 0:
                batch_feat = struct_feats[batch_mask].mean(dim=0)
            else:
                batch_feat = torch.zeros(hidden_dim, device=device)

            sample_mut_mask = batch_mask & mutation_mask
            anchor_mask = batch_mask & mutation_ca_mask
            if anchor_mask.sum() == 0:
                anchor_mask = sample_mut_mask

            if sample_mut_mask.sum() > 0:
                mutation_feat = struct_feats[sample_mut_mask].mean(dim=0)
            else:
                mutation_feat = torch.zeros(hidden_dim, device=device)

            local_mask = torch.zeros_like(batch_mask, dtype=torch.bool)
            if anchor_mask.sum() > 0 and batch_mask.sum() > 0:
                sample_indices = torch.nonzero(batch_mask, as_tuple=True)[0]
                distances = torch.cdist(coords[sample_indices], coords[anchor_mask])
                nearest_distance = distances.min(dim=1).values
                local_indices = sample_indices[nearest_distance <= self.local_radius]
                local_mask[local_indices] = True

            if local_mask.sum() > 0:
                local_feat = struct_feats[local_mask].mean(dim=0)
            else:
                local_feat = mutation_feat
            
            aggregated_feats.append(batch_feat)
            local_feats.append(torch.cat([mutation_feat, local_feat], dim=0))
        
        return torch.stack(aggregated_feats, dim=0), torch.stack(local_feats, dim=0)


class ESM_FoldX_DDAffinity(ESM_RAAD_FoldX_DDAffinity):
    
    def __init__(self, **kwargs):
        raad_params = ['raad_hidden_dim', 'raad_layers', 'edge_types', 
                       'rball_radius', 'knn_k', 'use_atom_features']
        for param in raad_params:
            kwargs.pop(param, None)
        
        super().__init__(** kwargs)
    
    def forward(self, antibody_seqs, antigen_seqs, mutant_antibody_seqs, mutant_antigen_seqs, foldx_energies,
                wt_esm_embedding=None, mut_esm_embedding=None, mutation_esm_embedding=None,
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
            foldx_features=foldx_features,
        )
