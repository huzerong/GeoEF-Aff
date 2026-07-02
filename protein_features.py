import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from Bio.PDB import PDBParser


class ProteinFeatureExtractor(nn.Module):
    
    def __init__(self, use_atom_features: bool = True):
        super().__init__()
        self.use_atom_features = use_atom_features
        
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'SE': 5, 
            'H': 6, 'CL': 7, 'BR': 8, 'I': 9, 'F': 10, 'UNK': 11
        }
        
        self.aa_types = {
            'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
            'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
            'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
            'UNK': 20
        }
        
        self.rbf_layer = RBFLayer(num_rbf=16, max_distance=20.0)
        
        # 修复特征维度计算错误（20 → 21，匹配aa_types的21种类型）
        feature_dim = 21 + 16 + 6  # 氨基酸one-hot(21) + RBF(16) + 几何特征(6)
        if use_atom_features:
            feature_dim += 12  # 原子特征(12)
        
        self.feature_proj = nn.Linear(feature_dim, 128)  # 输入维度与实际特征匹配
    
    def extract_from_pdb(self, pdb_path: str, chain_ids: Dict[str, str]) -> Dict[str, torch.Tensor]:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        features = {
            'coords': [],
            'aa_types': [],
            'atom_types': [],
            'segment_ids': [],
            'seq_idx': [],
            'chain_ids': [],
            'residue_names': [],
            'atom_names': []
        }
        
        segment_mapping = {'heavy': 1, 'light': 2, 'antigen': 0}
        
        for chain_name, chain_id in chain_ids.items():
            if chain_id not in [chain.id for chain in structure.get_chains()]:
                continue
                
            chain = structure[0][chain_id]
            segment_id = segment_mapping.get(chain_name, 0)
            
            for i, residue in enumerate(chain.get_residues()):
                if not self._is_standard_residue(residue):
                    continue
                
                atoms_to_extract = ['CA', 'C', 'N', 'O']
                residue_coords = []
                residue_atom_types = []
                residue_atom_names = []
                
                for atom_name in atoms_to_extract:
                    if atom_name in residue:
                        atom = residue[atom_name]
                        coord = atom.get_coord()
                        atom_type = self._get_atom_type(atom.element)
                        
                        residue_coords.append(coord)
                        residue_atom_types.append(atom_type)
                        residue_atom_names.append(atom_name)
                
                if len(residue_coords) >= 3:
                    while len(residue_coords) < 4:
                        residue_coords.append(residue_coords[0])
                        residue_atom_types.append(residue_atom_types[0])
                        residue_atom_names.append('DUM')
                    
                    aa_type = self._get_aa_type(residue.resname)
                    
                    features['coords'].extend(residue_coords)
                    features['aa_types'].extend([aa_type] * len(residue_coords))
                    features['atom_types'].extend(residue_atom_types)
                    features['segment_ids'].extend([segment_id] * len(residue_coords))
                    features['seq_idx'].extend([i] * len(residue_coords))
                    features['chain_ids'].extend([chain_id] * len(residue_coords))
                    features['residue_names'].extend([residue.resname] * len(residue_coords))
                    features['atom_names'].extend(residue_atom_names)
        
        for key in ['coords', 'aa_types', 'atom_types', 'segment_ids', 'seq_idx']:
            if features[key]:
                np_array = np.array(features[key])
                features[key] = torch.from_numpy(np_array).to(torch.float32 if key == 'coords' else torch.long)
            else:
                if key == 'coords':
                    features[key] = torch.empty(0, 3, dtype=torch.float32)
                else:
                    features[key] = torch.empty(0, dtype=torch.long)
        
        return features
    
    def _is_standard_residue(self, residue) -> bool:
        return residue.resname in self.aa_types and residue.id[0] == ' '
    
    def _get_atom_type(self, element: str) -> int:
        return self.atom_types.get(element.upper(), self.atom_types['UNK'])
    
    def _get_aa_type(self, resname: str) -> int:
        return self.aa_types.get(resname, self.aa_types['UNK'])
    
    def compute_geometric_features(self, coords: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        num_atoms = coords.shape[0]
        # 修复：确保几何特征与输入coords在同一设备
        geo_features = torch.zeros(num_atoms, 6, device=coords.device)
        
        for i in range(num_atoms):
            if i < num_atoms - 3:
                v1 = coords[i+1] - coords[i]
                v2 = coords[i+2] - coords[i+1]
                v3 = coords[i+3] - coords[i+2]
                
                bond_length_1 = torch.norm(v1)
                bond_length_2 = torch.norm(v2)
                
                bond_angle = torch.acos(torch.clamp(
                    torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2)), -1, 1
                ))
                
                # 修复：显式指定cross的dim参数，消除警告
                n1 = torch.cross(v1, v2, dim=0)
                n2 = torch.cross(v2, v3, dim=0)
                dihedral = torch.atan2(
                    torch.dot(torch.cross(n1, n2, dim=0), v2) / torch.norm(v2),
                    torch.dot(n1, n2)
                )
                
                # 修复：确保张量在正确设备上
                geo_features[i] = torch.tensor([
                    bond_length_1, bond_length_2, bond_angle,
                    torch.sin(dihedral), torch.cos(dihedral), 
                    float(segment_ids[i])
                ], device=coords.device)
        
        return geo_features
    
    def forward(
        self, 
        coords: torch.Tensor,
        aa_types: torch.Tensor,
        atom_types: torch.Tensor,
        segment_ids: torch.Tensor
    ) -> torch.Tensor:
        # 获取设备并确保所有特征在同一设备
        device = coords.device
        
        # 修复：氨基酸one-hot编码维度为21（匹配aa_types的21种类型）
        aa_onehot = F.one_hot(aa_types, num_classes=21).float().to(device)
        
        center_coords = coords.mean(dim=0, keepdim=True)
        distances = torch.norm(coords - center_coords, dim=1, keepdim=True)
        rbf_features = self.rbf_layer(distances).to(device)
        
        geo_features = self.compute_geometric_features(coords, segment_ids)  # 已在compute中处理设备
        
        features = [aa_onehot, rbf_features, geo_features]
        
        if self.use_atom_features:
            atom_onehot = F.one_hot(atom_types, num_classes=12).float().to(device)
            features.append(atom_onehot)
        
        # 拼接后的特征维度：21+16+6=43（不使用原子特征）或43+12=55（使用原子特征）
        node_features = torch.cat(features, dim=1)
        
        node_features = self.feature_proj(node_features)  # 维度匹配
        
        return node_features


class RBFLayer(nn.Module):
    
    def __init__(self, num_rbf: int = 16, max_distance: float = 20.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.max_distance = max_distance
        
        self.register_buffer(
            'centers', 
            torch.linspace(0, max_distance, num_rbf)
        )
        
        self.gamma = 1.0 / ((max_distance / num_rbf) ** 2)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.clamp(max=self.max_distance)
        if distances.dim() == 2 and distances.shape[1] == 1:
            distances = distances.squeeze(1)
        
        rbf_features = torch.exp(
            -self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2
        )
        return rbf_features


class SequenceAligner(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def align_sequence_to_structure(
        self, 
        sequence: str,
        structure_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_length = len(sequence)
        struct_length = len(structure_features['residue_names'])
        
        if seq_length != struct_length:
            print(f"Warning: Sequence length ({seq_length}) != Structure length ({struct_length})")
            min_length = min(seq_length, struct_length)
            sequence = sequence[:min_length]
            for key in structure_features:
                if isinstance(structure_features[key], torch.Tensor):
                    structure_features[key] = structure_features[key][:min_length]
                else:
                    structure_features[key] = structure_features[key][:min_length]
        
        aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20
        }
        
        seq_tensor = torch.tensor([aa_to_idx.get(aa, 20) for aa in sequence], dtype=torch.long)
        
        return seq_tensor, structure_features


def extract_cdr_regions(
    sequence: str, 
    structure_features: Dict[str, torch.Tensor],
    cdr_definitions: Optional[Dict[str, Tuple[int, int]]] = None
) -> torch.Tensor:
    seq_length = len(sequence)
    cdr_mask = torch.zeros(seq_length, dtype=torch.bool)
    
    if cdr_definitions:
        for cdr_name, (start, end) in cdr_definitions.items():
            if start < seq_length and end < seq_length:
                cdr_mask[start:end+1] = True
    
    return cdr_mask