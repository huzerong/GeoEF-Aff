import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_scatter import scatter_mean, scatter_add
except ImportError:
    def scatter_add(src, index, dim=0, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max()) + 1
        result = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        result.index_add_(dim, index, src)
        return result
    
    def scatter_mean(src, index, dim=0, dim_size=None):
        if dim_size is None:
            dim_size = int(index.max()) + 1
        sum_result = scatter_add(src, index, dim, dim_size)
        count = torch.zeros(dim_size, dtype=torch.float, device=src.device)
        ones = torch.ones_like(index, dtype=torch.float, device=src.device)
        count = count.scatter_add_(dim, index, ones)
        count = torch.clamp(count, min=1)
        if src.dim() > 1:
            count = count.view(-1, *([1] * (src.dim() - 1)))
        return sum_result / count
from typing import List, Tuple, Optional, Dict
import numpy as np
try:
    import pyro
    import pyro.distributions as dist
    HAS_PYRO = True
except ImportError:
    HAS_PYRO = False
    class RelaxedBernoulliStraightThrough:
        def __init__(self, temperature, probs):
            self.probs = probs
        
        def rsample(self):
            return (torch.rand_like(self.probs) < self.probs).float()


class RelationMPNN(nn.Module):
    
    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        n_layers: int,
        edge_type: int = 8,
        dropout: float = 0.0,
        act_fn: nn.Module = nn.ReLU(),
        coords_agg: str = 'mean',
        attention: bool = False,
        recurrent: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.input_nf = input_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.edge_type = edge_type
        self.coords_agg = coords_agg
        self.attention = attention
        self.recurrent = recurrent
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        
        self.relation_mlp = nn.ModuleList()
        self.coord_mlp = nn.ModuleList()
        for _ in range(edge_type):
            self.relation_mlp.append(nn.Linear(input_nf, input_nf, bias=False))
            self.coord_mlp.append(nn.Sequential(
                nn.Linear(input_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1)
            ))
        
        if self.recurrent:
            self.gru = nn.GRU(hidden_nf, input_nf)
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(input_nf + hidden_nf, hidden_nf),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(hidden_nf, output_nf)
            )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + 1, hidden_nf),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        
        if attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )
    
    def coord_model(
        self, 
        coord: torch.Tensor,
        edge_list: List[torch.Tensor],
        edge_feat_list: List[torch.Tensor],
        coord_diff_list: List[torch.Tensor],
        segment_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size = coord.shape[0]
        agg = torch.zeros_like(coord)
        
        for i, (edges, edge_feats, coord_diff) in enumerate(zip(edge_list, edge_feat_list, coord_diff_list)):
            if edges.shape[0] == 0:
                continue
                
            trans = coord_diff * self.coord_mlp[i](edge_feats).unsqueeze(-1)
            
            if i >= 6 and self.training:
                antigen_edge_list = (segment_ids[edges[:, 0]] == 0) & (segment_ids[edges[:, 1]] != 0)
                sampled_index = torch.ones(trans.shape[0]).to(trans.device)
                
                if antigen_edge_list.sum() != 0:
                    weight = torch.abs(self.coord_mlp[i](edge_feats).mean(dim=-1))[antigen_edge_list]
                    if weight.max() > weight.min():
                        probs = (weight - weight.min()) / (weight.max() - weight.min())
                        if HAS_PYRO:
                            sampled_index[antigen_edge_list] = pyro.distributions.RelaxedBernoulliStraightThrough(
                                temperature=0.5, probs=probs
                            ).rsample()
                        else:
                            sampled_index[antigen_edge_list] = RelaxedBernoulliStraightThrough(
                                temperature=0.5, probs=probs
                            ).rsample()
                
                trans = trans * sampled_index.unsqueeze(-1).unsqueeze(-1)
            
            if self.coords_agg == 'mean':
                agg.index_add_(0, edges[:, 1], trans.squeeze(1))
            else:
                agg = scatter_add(trans.squeeze(1), edges[:, 1], dim=0, dim_size=batch_size)
        
        agg = torch.clamp(agg, min=-10.0, max=10.0) # 限制幅度
        return coord + agg

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_list: List[torch.Tensor],
        segment_ids: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_feat_list = []
        coord_diff_list = []
        
        for edges in edge_list:
            if edges.shape[0] == 0:
                edge_feat_list.append(torch.empty(0, self.hidden_nf).to(h.device))
                coord_diff_list.append(torch.empty(0, 1, 3).to(coord.device))
                continue
            
            row, col = edges[:, 0], edges[:, 1]
            coord_diff = coord[row] - coord[col]
            radial = torch.sqrt(torch.sum(coord_diff ** 2, dim=1, keepdim=True))
            
            edge_input = torch.cat([h[row], h[col], radial], dim=1)
            edge_feat = self.edge_mlp(edge_input)
            
            if self.attention:
                att_val = self.att_mlp(edge_feat)
                edge_feat = edge_feat * att_val
            
            edge_feat_list.append(edge_feat)
            coord_diff_list.append(coord_diff.unsqueeze(1))
        
        coord = self.coord_model(coord, edge_list, edge_feat_list, coord_diff_list, segment_ids)
        
        agg_feat = torch.zeros_like(h)
        
        for i, (edges, edge_feats) in enumerate(zip(edge_list, edge_feat_list)):
            if edges.shape[0] == 0:
                continue
            
            messages = self.relation_mlp[i](edge_feats)
            
            agg_feat.index_add_(0, edges[:, 1], messages)
        
        if self.recurrent:
            agg_feat = agg_feat.unsqueeze(0) # GRU
            h = h.unsqueeze(0)
            _, h = self.gru(agg_feat, h)
            h = h.squeeze(0)
        else:
            h_in = h
            h_input = torch.cat([h, agg_feat], dim=1)
            h = self.node_mlp(h_input)

            if self.residual:
                h = h + h_in
        
        return h, coord


class RelationEGNN(nn.Module):
    
    def __init__(
        self,
        input_nf: int,
        hidden_nf: int,
        output_nf: int,
        n_layers: int,
        edge_type: int = 8,
        dropout: float = 0.0,
        act_fn: nn.Module = nn.ReLU(),
        coords_agg: str = 'mean',
        attention: bool = False,
        residual: bool = True,
    ):
        super().__init__()
        self.input_nf = input_nf
        self.hidden_nf = hidden_nf  
        self.output_nf = output_nf
        self.n_layers = n_layers
        self.edge_type = edge_type
        
        self.embedding = nn.Linear(input_nf, hidden_nf)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                RelationMPNN(
                    input_nf=hidden_nf,
                    output_nf=hidden_nf,
                    hidden_nf=hidden_nf,
                    n_layers=1,
                    edge_type=edge_type,
                    dropout=dropout,
                    act_fn=act_fn,
                    coords_agg=coords_agg,
                    attention=attention,
                    residual=residual,
                )
            )
        
        self.output_proj = nn.Linear(hidden_nf, output_nf)
    
    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_list: List[torch.Tensor],
        segment_ids: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embedding(h)
        
        for layer in self.layers:
            h, coord = layer(h, coord, edge_list, segment_ids, node_mask)
        
        h = self.output_proj(h)
        
        return h, coord


class EdgeConstructor(nn.Module): # GNN KD tree

    def __init__(self, rball_radius: float = 10.0, knn_k: int = 10):
        super().__init__()
        self.rball_radius = rball_radius
        self.knn_k = knn_k

    def build_edges(
        self,
        coord: torch.Tensor,
        segment_ids: torch.Tensor,
        batch_id: torch.Tensor,
        seq_idx: torch.Tensor
    ) -> List[torch.Tensor]:
        from torch_cluster import radius_graph, knn_graph

        edge_list = []
        device = coord.device

        # --- 1. Intra-chain radius ball edges (same segment) ---
        ctx_edges_rball = self._build_rball_edges_cluster(
            coord, segment_ids, batch_id, self.rball_radius, same_segment=True)
        edge_list.append(ctx_edges_rball)

        # --- 2,3. Global placeholder edges ---
        global_normal = torch.empty(0, 2, dtype=torch.long, device=device)
        global_global = torch.empty(0, 2, dtype=torch.long, device=device)
        edge_list.extend([global_normal, global_global])

        # --- 4,5. Sequential edges (d=1, d=2) ---
        ctx_edges_seq_d1 = self._build_sequential_edges(seq_idx, segment_ids, batch_id, distance=1)
        ctx_edges_seq_d2 = self._build_sequential_edges(seq_idx, segment_ids, batch_id, distance=2)
        edge_list.extend([ctx_edges_seq_d1, ctx_edges_seq_d2])

        # --- 6. Intra-chain KNN edges ---
        ctx_edges_knn = self._build_knn_edges_cluster(
            coord, segment_ids, batch_id, self.knn_k, same_segment=True)
        edge_list.append(ctx_edges_knn)

        # --- 7,8. Inter-chain radius ball + KNN edges ---
        inter_edges_rball = self._build_rball_edges_cluster(
            coord, segment_ids, batch_id, self.rball_radius, same_segment=False)
        inter_edges_knn = self._build_knn_edges_cluster(
            coord, segment_ids, batch_id, self.knn_k, same_segment=False)
        edge_list.extend([inter_edges_rball, inter_edges_knn])

        return edge_list

    def _build_rball_edges_cluster(
        self,
        coord: torch.Tensor,
        segment_ids: torch.Tensor,
        batch_id: torch.Tensor,
        radius: float,
        same_segment: bool = True,
    ) -> torch.Tensor:
        """用 torch_cluster.radius_graph 替代 N×N 距离矩阵，O(N·K) 复杂度"""
        from torch_cluster import radius_graph

        # radius_graph 返回 [2, E] 格式的边
        edge_index = radius_graph(coord, r=radius, batch=batch_id, loop=False,
                                  max_num_neighbors=64)
        row, col = edge_index[0], edge_index[1]

        # 按 same_segment 过滤
        if same_segment:
            mask = segment_ids[row] == segment_ids[col]
        else:
            mask = segment_ids[row] != segment_ids[col]

        edges = torch.stack([row[mask], col[mask]], dim=1)
        if edges.shape[0] == 0:
            return torch.empty(0, 2, dtype=torch.long, device=coord.device)
        return edges

    def _build_knn_edges_cluster(
        self,
        coord: torch.Tensor,
        segment_ids: torch.Tensor,
        batch_id: torch.Tensor,
        k: int,
        same_segment: bool = True,
    ) -> torch.Tensor:
        """用 torch_cluster.knn_graph 替代 N×N 距离矩阵"""
        from torch_cluster import knn_graph

        edge_index = knn_graph(coord, k=k, batch=batch_id, loop=False)
        row, col = edge_index[0], edge_index[1]

        if same_segment:
            mask = segment_ids[row] == segment_ids[col]
        else:
            mask = segment_ids[row] != segment_ids[col]

        edges = torch.stack([row[mask], col[mask]], dim=1)
        if edges.shape[0] == 0:
            return torch.empty(0, 2, dtype=torch.long, device=coord.device)
        return edges
    
    def _build_sequential_edges(
        self,
        seq_idx: torch.Tensor,
        segment_ids: torch.Tensor,
        batch_id: torch.Tensor,
        distance: int = 1
    ) -> torch.Tensor:
        edges = []
        
        for sample_id in batch_id.unique():
            sample_mask = batch_id == sample_id
            for segment_id in segment_ids[sample_mask].unique():
                segment_mask = sample_mask & (segment_ids == segment_id)
                segment_positions = torch.nonzero(segment_mask, as_tuple=True)[0]
                segment_seq_idx = seq_idx[segment_mask]

                sorted_indices = torch.argsort(segment_seq_idx)
                sorted_positions = segment_positions[sorted_indices]

                for i in range(len(sorted_positions) - distance):
                    edges.append([sorted_positions[i], sorted_positions[i + distance]])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).to(seq_idx.device)
        else:
            return torch.empty(0, 2, dtype=torch.long).to(seq_idx.device)
