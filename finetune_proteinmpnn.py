#!/usr/bin/env python3

"""
ProteinMPNN Fine-tuning Script with LoRA, Layer Freezing, and EWC Support

This version adds three optional features to prevent catastrophic forgetting:
1. LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
2. Layer freezing to preserve pretrained representations
3. Elastic Weight Consolidation (EWC) to protect important weights

New arguments:
--use_lora: Enable LoRA adapters
--lora_rank: Rank for LoRA matrices (default: 8)
--lora_alpha: LoRA scaling parameter (default: 16)
--freeze_encoder_layers: Number of encoder layers to freeze from the start (default: 0)
--freeze_decoder_layers: Number of decoder layers to freeze from the start (default: 0)
--use_ewc: Enable Elastic Weight Consolidation
--ewc_lambda: EWC regularization strength (default: 1000)
--ewc_sample_size: Number of samples for Fisher computation (default: 200)
"""

import os
import sys
import json
import argparse
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

try:
    from torch.cuda.amp import GradScaler, autocast
    HAS_AMP = True
except ImportError:
    HAS_AMP = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ==============================================================================
# LORA IMPLEMENTATION
# ==============================================================================

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer
    Adds trainable low-rank matrices A and B to a frozen linear layer
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original layer output + LoRA adaptation
        result = self.original_layer(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return result + lora_out


def add_lora_to_linear(module, rank=8, alpha=16, target_modules=None):
    """
    Recursively add LoRA to linear layers in a module

    Args:
        module: The module to modify
        rank: Rank of LoRA matrices
        alpha: LoRA scaling parameter
        target_modules: List of module name patterns to target (e.g., ['W1', 'W2', 'W3'])
    """
    if target_modules is None:
        target_modules = ['W1', 'W2', 'W3', 'W11', 'W12', 'W13', 'W_in', 'W_out']

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
            # Replace with LoRA version
            lora_layer = LoRALayer(child, rank=rank, alpha=alpha)
            setattr(module, name, lora_layer)
        else:
            # Recurse
            add_lora_to_linear(child, rank, alpha, target_modules)


def count_lora_parameters(model):
    """Count trainable LoRA parameters"""
    lora_params = sum(p.numel() for n, p in model.named_parameters() 
                     if p.requires_grad and ('lora_A' in n or 'lora_B' in n))
    total_params = sum(p.numel() for p in model.parameters())
    return lora_params, total_params

# ==============================================================================
# EWC IMPLEMENTATION
# ==============================================================================

class EWC:
    """
    Elastic Weight Consolidation
    Computes Fisher Information Matrix and adds regularization to preserve important weights
    """
    def __init__(self, model, dataset, device, sample_size=200):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataset, sample_size)

    def _compute_fisher(self, dataset, sample_size):
        """Compute Fisher Information Matrix using sampled data"""
        print(f"Computing Fisher Information Matrix on {sample_size} samples...")
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.eval()

        # Sample indices
        indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)

        for idx in indices:
            self.model.zero_grad()

            # Get a single sample
            sample = dataset[idx]

            # Create batch of size 1
            batch = {
                'X': sample['X'].unsqueeze(0).to(self.device),
                'S': sample['S'].unsqueeze(0).to(self.device),
                'mask': sample['mask'].unsqueeze(0).to(self.device),
                'chain_M': sample['chain_M'].unsqueeze(0).to(self.device),
                'residue_idx': sample['residue_idx'].unsqueeze(0).to(self.device),
                'chain_encoding_all': sample['chain_encoding_all'].unsqueeze(0).to(self.device)
            }

            randn = torch.randn(1, batch['X'].shape[1], device=self.device)

            # Forward pass
            log_probs = self.model(
                batch['X'], batch['S'], batch['mask'], batch['chain_M'],
                batch['residue_idx'], batch['chain_encoding_all'], randn
            )

            # Compute loss
            loss_mask = batch['chain_M'] * batch['mask']
            _, loss = loss_nll(batch['S'], log_probs, loss_mask)

            # Backward to get gradients
            loss.backward()

            # Accumulate squared gradients (Fisher = E[grad^2])
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        # Normalize by sample size
        for n in fisher:
            fisher[n] /= sample_size

        print("Fisher Information Matrix computed successfully!")
        return fisher

    def penalty(self, model):
        """Compute EWC penalty term"""
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss

# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ==============================================================================
# MODEL COMPONENTS - EXACT MATCH TO ORIGINAL PROTEINMPNN
# ==============================================================================

def gather_edges(edges, neighbor_idx):
    """Features [B,N,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]"""
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]"""
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)
        
    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask + \
            (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class EncLayer(nn.Module):
    """Encoder layer - EXACTLY matches original ProteinMPNN"""
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""
        # Message passing for nodes
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        
        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
            
        # Message passing for edges
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        
        return h_V, h_E


class DecLayer(nn.Module):
    """Decoder layer - EXACTLY matches original ProteinMPNN"""
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        
    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        
        h_V = self.norm1(h_V + self.dropout1(dh))
        
        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class ProteinFeatures(nn.Module):
    """Extract protein features - EXACTLY matches original ProteinMPNN"""
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
                 num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        super().__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Edge input: positional (16) + RBF for 25 atom pairs (16*25=400) = 416
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        
    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx
    
    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF
    
    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :])**2, -1) + 1e-6)
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B
    
    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            
        # Calculate virtual Cb
        b = X[:, :, 1, :] - X[:, :, 0, :]  # CA - N
        c = X[:, :, 2, :] - X[:, :, 1, :]  # C - CA
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
        
        D_neighbors, E_idx = self._dist(Ca, mask)
        
        # 25 RBF features for all atom pairs
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))      # Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))   # N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))   # C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))   # O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) # Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  # Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  # Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  # Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) # Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))   # N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))   # N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  # N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  # Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  # Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))   # O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  # N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  # C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  # O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) # Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))   # C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))   # O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  # Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  # C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  # O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))   # C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        
        # Positional encodings
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        
        return E, E_idx


class ProteinMPNN(nn.Module):
    """ProteinMPNN model - EXACTLY matches original architecture"""
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=64, augment_eps=0.05, dropout=0.1, ca_only=False):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Feature extraction (full backbone, not CA-only)
        self.features = ProteinFeatures(
            node_features, edge_features, 
            top_k=k_neighbors, augment_eps=augment_eps
        )
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        
        # Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn, 
                use_input_decoding_order=False, decoding_order=None):
        """Graph-conditioned sequence model - teacher forcing for training"""
        device = X.device
        
        # Get edge features and indices
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)
        
        # Encoder (unmasked self-attention)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            
        # Sequence embeddings
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        
        # Build encoder embeddings for decoder
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        
        # Update chain mask with missing regions
        chain_M = chain_M * mask
        
        # Decoding order
        if not use_input_decoding_order:
            decoding_order = torch.argsort((chain_M + 0.0001) * torch.abs(randn))
            
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
            
        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs


# ==============================================================================
# DATASET
# ==============================================================================

class ProteinDataset(Dataset):
    """Dataset for protein structures with fixed positions support"""
    
    def __init__(self, data_dir, max_length=2000, 
                 fixed_chains_dict=None, fixed_positions_dict=None, verbose=True):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.fixed_chains_dict = fixed_chains_dict or {}
        self.fixed_positions_dict = fixed_positions_dict or {}
        self.verbose = verbose
        
        self.alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.alphabet)}
        
        self.data = self._load_data()
        if verbose:
            print(f"Loaded {len(self.data)} structures")
            
    def _load_data(self):
        data = []
        
        # Check for JSONL files
        jsonl_files = list(self.data_dir.glob("*.jsonl"))
        if jsonl_files:
            for f in jsonl_files:
                data.extend(self._load_jsonl(f))
        else:
            # Try JSON files
            json_files = list(self.data_dir.glob("*.json"))
            for f in json_files:
                data.extend(self._load_json(f))
                
        if not data:
            # Try PDB files
            pdb_files = list(self.data_dir.glob("*.pdb"))
            for f in pdb_files:
                entry = self._parse_pdb(f)
                if entry:
                    data.append(entry)
                    
        return data
    
    def _load_jsonl(self, filepath):
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                data.append(entry)
        return data
    
    def _load_json(self, filepath):
        with open(filepath, 'r') as f:
            content = json.load(f)
        if isinstance(content, list):
            return content
        return [content]
    
    def _parse_pdb(self, pdb_path):
        """Simple PDB parser"""
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        residue_data = {}
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    resname = line[17:20].strip()
                    chain_id = line[21]
                    res_num = line[22:27].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    if resname not in three_to_one:
                        continue
                        
                    key = (chain_id, res_num, resname)
                    if key not in residue_data:
                        residue_data[key] = {'order': len(residue_data)}
                    if atom_name in ['N', 'CA', 'C', 'O']:
                        residue_data[key][atom_name] = [x, y, z]
                        
        # Build output
        result = {'name': pdb_path.stem, 'seq': ''}
        chains = {}
        
        sorted_residues = sorted(residue_data.items(), key=lambda x: x[1]['order'])
        
        for (chain_id, res_num, resname), atoms in sorted_residues:
            if all(a in atoms for a in ['N', 'CA', 'C', 'O']):
                if chain_id not in chains:
                    chains[chain_id] = {'coords': {'N': [], 'CA': [], 'C': [], 'O': []}, 'seq': ''}
                for atom in ['N', 'CA', 'C', 'O']:
                    chains[chain_id]['coords'][atom].append(atoms[atom])
                chains[chain_id]['seq'] += three_to_one[resname]
                
        for chain_id in sorted(chains.keys()):
            result[f'seq_chain_{chain_id}'] = chains[chain_id]['seq']
            result[f'coords_chain_{chain_id}'] = {
                f'N_chain_{chain_id}': chains[chain_id]['coords']['N'],
                f'CA_chain_{chain_id}': chains[chain_id]['coords']['CA'],
                f'C_chain_{chain_id}': chains[chain_id]['coords']['C'],
                f'O_chain_{chain_id}': chains[chain_id]['coords']['O'],
            }
            result['seq'] += chains[chain_id]['seq']
            
        if not result['seq']:
            return None
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        name = entry.get('name', f'protein_{idx}')
        
        # Collect all chains
        all_coords = []
        all_seq = []
        chain_encoding = []
        residue_idx = []
        chain_mask = []
        
        chain_offset = 0
        chain_idx = 1
        
        # Find all chains in entry
        chain_keys = sorted([k.split('_')[-1] for k in entry.keys() if k.startswith('seq_chain_')])
        
        for chain_id in chain_keys:
            seq_key = f'seq_chain_{chain_id}'
            coords_key = f'coords_chain_{chain_id}'
            
            if seq_key not in entry or coords_key not in entry:
                continue
                
            chain_seq = entry[seq_key].replace('-', 'X')
            chain_coords_dict = entry[coords_key]
            
            # Get coordinates
            N_coords = chain_coords_dict.get(f'N_chain_{chain_id}', [])
            CA_coords = chain_coords_dict.get(f'CA_chain_{chain_id}', [])
            C_coords = chain_coords_dict.get(f'C_chain_{chain_id}', [])
            O_coords = chain_coords_dict.get(f'O_chain_{chain_id}', [])
            
            L_chain = len(CA_coords)
            if L_chain == 0:
                continue
                
            # Stack coordinates [L, 4, 3]
            coords = np.stack([N_coords, CA_coords, C_coords, O_coords], axis=1)
            
            all_coords.append(coords)
            all_seq.extend(chain_seq)
            chain_encoding.extend([chain_idx] * L_chain)
            residue_idx.extend([100 * (chain_idx - 1) + i for i in range(L_chain)])
            
            # Determine fixed positions
            fixed_chains = self.fixed_chains_dict.get(name, [])
            fixed_pos = self.fixed_positions_dict.get(name, {}).get(chain_id, [])
            
            for i in range(L_chain):
                if chain_id in fixed_chains:
                    chain_mask.append(0)
                elif (i + 1) in fixed_pos:
                    chain_mask.append(0)
                else:
                    chain_mask.append(1)
                    
            chain_offset += L_chain
            chain_idx += 1
            
        if not all_coords:
            # Return dummy data if parsing failed
            return self.__getitem__((idx + 1) % len(self.data))
            
        all_coords = np.concatenate(all_coords, axis=0)
        
        # Truncate if needed
        L = len(all_seq)
        if L > self.max_length:
            all_coords = all_coords[:self.max_length]
            all_seq = all_seq[:self.max_length]
            chain_encoding = chain_encoding[:self.max_length]
            residue_idx = residue_idx[:self.max_length]
            chain_mask = chain_mask[:self.max_length]
            L = self.max_length
            
        # Convert sequence to indices
        seq_idx = [self.aa_to_idx.get(aa, 20) for aa in all_seq]
        
        # Create mask for valid positions
        mask = np.isfinite(np.sum(all_coords, axis=(1, 2))).astype(np.float32)
        all_coords = np.nan_to_num(all_coords)
        
        return {
            'name': name,
            'X': torch.tensor(all_coords, dtype=torch.float32),
            'S': torch.tensor(seq_idx, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'chain_M': torch.tensor(chain_mask, dtype=torch.float32),
            'residue_idx': torch.tensor(residue_idx, dtype=torch.long),
            'chain_encoding_all': torch.tensor(chain_encoding, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function with padding"""
    max_len = max(item['X'].shape[0] for item in batch)
    
    padded = {k: [] for k in ['X', 'S', 'mask', 'chain_M', 'residue_idx', 'chain_encoding_all']}
    names = []
    
    for item in batch:
        L = item['X'].shape[0]
        pad_len = max_len - L
        names.append(item['name'])
        
        padded['X'].append(F.pad(item['X'], (0, 0, 0, 0, 0, pad_len)))
        padded['S'].append(F.pad(item['S'], (0, pad_len), value=20))
        padded['mask'].append(F.pad(item['mask'], (0, pad_len), value=0))
        padded['chain_M'].append(F.pad(item['chain_M'], (0, pad_len), value=0))
        padded['residue_idx'].append(F.pad(item['residue_idx'], (0, pad_len), value=0))
        padded['chain_encoding_all'].append(F.pad(item['chain_encoding_all'], (0, pad_len), value=0))
        
    return {
        'names': names,
        'X': torch.stack(padded['X']),
        'S': torch.stack(padded['S']),
        'mask': torch.stack(padded['mask']),
        'chain_M': torch.stack(padded['chain_M']),
        'residue_idx': torch.stack(padded['residue_idx']),
        'chain_encoding_all': torch.stack(padded['chain_encoding_all'])
    }



# ==============================================================================
# TRAINING
# ==============================================================================

def loss_nll(S, log_probs, mask):
    """Negative log likelihood loss"""
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """Label smoothed loss"""
    S_onehot = F.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def freeze_layers(model, freeze_encoder_layers=0, freeze_decoder_layers=0):
    """
    Freeze specified number of encoder and decoder layers

    Args:
        model: ProteinMPNN model
        freeze_encoder_layers: Number of encoder layers to freeze (from start)
        freeze_decoder_layers: Number of decoder layers to freeze (from start)
    """
    frozen_params = 0
    total_params = 0

    # Freeze encoder layers
    if freeze_encoder_layers > 0:
        for i in range(min(freeze_encoder_layers, len(model.encoder_layers))):
            for param in model.encoder_layers[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            total_params += sum(p.numel() for p in model.encoder_layers[i].parameters())

    # Freeze decoder layers
    if freeze_decoder_layers > 0:
        for i in range(min(freeze_decoder_layers, len(model.decoder_layers))):
            for param in model.decoder_layers[i].parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            total_params += sum(p.numel() for p in model.decoder_layers[i].parameters())

    return frozen_params, total_params

def merge_lora_weights(model):
    """
    Merge LoRA weights back into original weights for vanilla ProteinMPNN compatibility
    
    Returns:
        merged_state_dict: State dict with LoRA merged into original weights
    """
    state_dict = model.state_dict()
    merged_state = {}
    processed_keys = set()
    
    print("Merging LoRA weights into original parameters...")
    
    # Process all keys
    for key in state_dict.keys():
        if 'lora_A' in key or 'lora_B' in key:
            # Skip LoRA-specific keys
            processed_keys.add(key)
            continue
        elif 'original_layer' in key:
            # This is a frozen weight inside LoRA layer
            new_key = key.replace('.original_layer', '')
            merged_state[new_key] = state_dict[key].clone()
            
            # Check for corresponding LoRA matrices
            base_module_key = new_key.rsplit('.', 1)[0]
            lora_A_key = base_module_key + '.lora_A'
            lora_B_key = base_module_key + '.lora_B'
            
            if lora_A_key in state_dict and lora_B_key in state_dict and 'weight' in new_key:
                lora_A = state_dict[lora_A_key]  # [in_features, rank]
                lora_B = state_dict[lora_B_key]  # [rank, out_features]
                
                rank = lora_A.shape[1]
                alpha = 16  # Default alpha from LoRA
                scaling = alpha / rank
                
                # Merge: W_new = W_original + scaling * (A @ B).T
                # Fixed: lora_A @ lora_B, not lora_B @ lora_A
                lora_weight = (lora_A @ lora_B).T  # [out_features, in_features]
                merged_state[new_key] = merged_state[new_key] + scaling * lora_weight
                
                print(f"  ✓ Merged LoRA into: {new_key} (rank={rank})")
            
            processed_keys.add(key)
        else:
            # Regular parameter
            merged_state[key] = state_dict[key].clone()
            processed_keys.add(key)
    
    lora_param_count = sum(1 for k in state_dict if 'lora' in k)
    print(f"Merge complete! Removed {lora_param_count} LoRA tensors")
    return merged_state



class Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, config=None,
                 output_dir="./outputs", device="cuda", ewc=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.device = device
        self.ewc = ewc  # EWC object if enabled

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        self.logger = setup_logging(str(self.output_dir))

        # Training params
        self.batch_size = config.get('batch_size', 8)
        self.num_epochs = config.get('num_epochs', 100)
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.use_amp = config.get('use_amp', True) and HAS_AMP and device == "cuda"
        self.label_smoothing = config.get('label_smoothing', 0.0)

        # EWC params
        self.use_ewc = config.get('use_ewc', False)
        self.ewc_lambda = config.get('ewc_lambda', 1000.0)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=config.get('num_workers', 4), collate_fn=collate_fn,
            pin_memory=True, drop_last=True
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=config.get('num_workers', 4), collate_fn=collate_fn,
                pin_memory=True
            )
        else:
            self.val_loader = None

        # Optimizer (only optimize trainable parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Scheduler
        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=self.lr * 0.01)

        # AMP
        self.scaler = GradScaler() if self.use_amp else None

        # Monitoring
        self.global_step = 0
        self.best_val_loss = float('inf')

        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        else:
            self.writer = None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_ewc_loss = 0
        total_task_loss = 0
        total_tokens = 0
        total_correct = 0

        for batch_idx, batch in enumerate(self.train_loader):
            X = batch['X'].to(self.device)
            S = batch['S'].to(self.device)
            mask = batch['mask'].to(self.device)
            chain_M = batch['chain_M'].to(self.device)
            residue_idx = batch['residue_idx'].to(self.device)
            chain_encoding = batch['chain_encoding_all'].to(self.device)
            randn = torch.randn(X.shape[0], X.shape[1], device=self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    log_probs = self.model(X, S, mask, chain_M, residue_idx, chain_encoding, randn)
                    loss_mask = chain_M * mask

                    if self.label_smoothing > 0:
                        _, task_loss = loss_smoothed(S, log_probs, loss_mask, self.label_smoothing)
                    else:
                        _, task_loss = loss_nll(S, log_probs, loss_mask)

                    # Add EWC penalty if enabled
                    if self.use_ewc and self.ewc is not None:
                        ewc_loss = self.ewc.penalty(self.model)
                        loss = task_loss + self.ewc_lambda * ewc_loss
                    else:
                        loss = task_loss
                        ewc_loss = torch.tensor(0.0)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_probs = self.model(X, S, mask, chain_M, residue_idx, chain_encoding, randn)
                loss_mask = chain_M * mask

                if self.label_smoothing > 0:
                    _, task_loss = loss_smoothed(S, log_probs, loss_mask, self.label_smoothing)
                else:
                    _, task_loss = loss_nll(S, log_probs, loss_mask)

                # Add EWC penalty if enabled
                if self.use_ewc and self.ewc is not None:
                    ewc_loss = self.ewc.penalty(self.model)
                    loss = task_loss + self.ewc_lambda * ewc_loss
                else:
                    loss = task_loss
                    ewc_loss = torch.tensor(0.0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.scheduler.step()

            # Metrics
            with torch.no_grad():
                predictions = log_probs.argmax(dim=-1)
                correct = ((predictions == S) * loss_mask).sum().item()
                total_correct += correct
                total_tokens += loss_mask.sum().item()
                total_loss += loss.item() * loss_mask.sum().item()
                total_task_loss += task_loss.item() * loss_mask.sum().item()
                if isinstance(ewc_loss, torch.Tensor):
                    total_ewc_loss += ewc_loss.item() * loss_mask.sum().item()

            self.global_step += 1

            if self.global_step % self.config.get('log_interval', 10) == 0:
                if self.writer:
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/task_loss', task_loss.item(), self.global_step)
                    if isinstance(ewc_loss, torch.Tensor) and ewc_loss.item() > 0:
                        self.writer.add_scalar('train/ewc_loss', ewc_loss.item(), self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        metrics = {
            'loss': total_loss / max(total_tokens, 1),
            'task_loss': total_task_loss / max(total_tokens, 1),
            'accuracy': total_correct / max(total_tokens, 1),
            'perplexity': np.exp(total_task_loss / max(total_tokens, 1))
        }

        if self.use_ewc:
            metrics['ewc_loss'] = total_ewc_loss / max(total_tokens, 1)

        return metrics

    @torch.no_grad()
    def evaluate(self):
        if not self.val_loader:
            return {}

        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_correct = 0

        for batch in self.val_loader:
            X = batch['X'].to(self.device)
            S = batch['S'].to(self.device)
            mask = batch['mask'].to(self.device)
            chain_M = batch['chain_M'].to(self.device)
            residue_idx = batch['residue_idx'].to(self.device)
            chain_encoding = batch['chain_encoding_all'].to(self.device)
            randn = torch.randn(X.shape[0], X.shape[1], device=self.device)

            log_probs = self.model(X, S, mask, chain_M, residue_idx, chain_encoding, randn)
            loss_mask = chain_M * mask
            _, loss = loss_nll(S, log_probs, loss_mask)

            predictions = log_probs.argmax(dim=-1)
            correct = ((predictions == S) * loss_mask).sum().item()

            total_loss += loss.item() * loss_mask.sum().item()
            total_tokens += loss_mask.sum().item()
            total_correct += correct

        return {
            'loss': total_loss / max(total_tokens, 1),
            'accuracy': total_correct / max(total_tokens, 1),
            'perplexity': np.exp(total_loss / max(total_tokens, 1))
        }

    def save_checkpoint(self, epoch, metrics=None, is_best=False):
        # Check if model has LoRA
        has_lora = any('lora_A' in name or 'lora_B' in name 
                    for name, _ in self.model.named_parameters())
        
        # Save regular checkpoint (with LoRA if present)
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'num_edges': self.config.get('k_neighbors', 48),
            'has_lora': has_lora
        }
        
        # Save epoch checkpoint
        torch.save(checkpoint, self.output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, self.output_dir / "checkpoints" / "checkpoint_latest.pt")
        
        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoints" / "checkpoint_best.pt")
            self.logger.info(f"New best model saved (val_loss={metrics.get('val_loss', 'N/A'):.4f})")
        
        # If using LoRA, also save merged version
        if has_lora:
            self.logger.info("Creating merged checkpoint (vanilla ProteinMPNN compatible)...")
            merged_state = merge_lora_weights(self.model)
            
            checkpoint_merged = checkpoint.copy()
            checkpoint_merged['model_state_dict'] = merged_state
            checkpoint_merged['has_lora'] = False
            
            # Save merged versions
            torch.save(checkpoint_merged, 
                    self.output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}_merged.pt")
            torch.save(checkpoint_merged, 
                    self.output_dir / "checkpoints" / "checkpoint_latest_merged.pt")
            
            if is_best:
                torch.save(checkpoint_merged, 
                        self.output_dir / "checkpoints" / "checkpoint_best_merged.pt")
                self.logger.info("✓ Saved checkpoint_best_merged.pt (use this for vanilla ProteinMPNN)")


    def train(self, resume_from=None):
        start_epoch = 0

        if resume_from:
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            self.global_step = ckpt.get('global_step', 0)
            self.logger.info(f"Resumed from epoch {start_epoch}")

        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset) if self.val_dataset else 0}")

        if self.use_ewc:
            self.logger.info(f"EWC enabled with lambda={self.ewc_lambda}")

        for epoch in range(start_epoch, self.num_epochs):
            train_metrics = self.train_epoch(epoch)

            log_msg = f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
            if 'ewc_loss' in train_metrics:
                log_msg += f", EWC: {train_metrics['ewc_loss']:.4f}"
            self.logger.info(log_msg)

            if self.val_loader and (epoch + 1) % self.config.get('eval_interval', 1) == 0:
                val_metrics = self.evaluate()
                self.logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

                if self.writer:
                    self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)

                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
            else:
                val_metrics = {}
                is_best = False

            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                metrics = {'train_loss': train_metrics['loss']}
                if val_metrics:
                    metrics['val_loss'] = val_metrics['loss']
                self.save_checkpoint(epoch, metrics, is_best)

        self.logger.info("Training complete!")
        if self.writer:
            self.writer.close()

# ==============================================================================
# TESTING
# ==============================================================================

@torch.no_grad()
def run_test(model, test_dataset, config, output_dir, device):
    """Run evaluation on test set and save detailed results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir))
    logger.info(f"Running test evaluation on {len(test_dataset)} structures")
    
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    model.eval()
    
    # Aggregate metrics
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    
    # Per-protein metrics
    per_protein_results = []
    
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    
    for batch_idx, batch in enumerate(test_loader):
        X = batch['X'].to(device)
        S = batch['S'].to(device)
        mask = batch['mask'].to(device)
        chain_M = batch['chain_M'].to(device)
        residue_idx = batch['residue_idx'].to(device)
        chain_encoding = batch['chain_encoding_all'].to(device)
        names = batch['names']
        
        randn = torch.randn(X.shape[0], X.shape[1], device=device)
        
        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding, randn)
        
        loss_mask = chain_M * mask
        loss_per_pos, _ = loss_nll(S, log_probs, loss_mask)
        
        predictions = log_probs.argmax(dim=-1)
        correct_per_pos = (predictions == S).float() * loss_mask
        
        # Aggregate
        batch_loss = (loss_per_pos * loss_mask).sum().item()
        batch_tokens = loss_mask.sum().item()
        batch_correct = correct_per_pos.sum().item()
        
        total_loss += batch_loss
        total_tokens += batch_tokens
        total_correct += batch_correct
        
        # Per-protein results
        for i in range(X.shape[0]):
            protein_mask = loss_mask[i]
            protein_tokens = protein_mask.sum().item()
            
            if protein_tokens > 0:
                protein_loss = (loss_per_pos[i] * protein_mask).sum().item() / protein_tokens
                protein_correct = correct_per_pos[i].sum().item()
                protein_acc = protein_correct / protein_tokens
                protein_ppl = np.exp(protein_loss)
                
                # Get predicted and true sequences (only for designable positions)
                pred_seq = ''.join([alphabet[idx] for idx, m in zip(predictions[i].cpu().numpy(), protein_mask.cpu().numpy()) if m > 0])
                true_seq = ''.join([alphabet[idx] for idx, m in zip(S[i].cpu().numpy(), protein_mask.cpu().numpy()) if m > 0])
                
                per_protein_results.append({
                    'name': names[i],
                    'length': int(protein_tokens),
                    'loss': protein_loss,
                    'accuracy': protein_acc,
                    'perplexity': protein_ppl,
                    'correct_residues': int(protein_correct),
                    'true_sequence': true_seq,
                    'predicted_sequence': pred_seq
                })
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate aggregate metrics
    avg_loss = total_loss / max(total_tokens, 1)
    avg_accuracy = total_correct / max(total_tokens, 1)
    avg_perplexity = np.exp(avg_loss)
    
    # Summary statistics
    results_summary = {
        'aggregate': {
            'total_proteins': len(per_protein_results),
            'total_residues': int(total_tokens),
            'total_correct': int(total_correct),
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'avg_perplexity': avg_perplexity
        },
        'per_protein': per_protein_results
    }
    
    # Log summary
    logger.info("=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total proteins:    {len(per_protein_results)}")
    logger.info(f"Total residues:    {int(total_tokens)}")
    logger.info(f"Average Loss:      {avg_loss:.4f}")
    logger.info(f"Average Accuracy:  {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    logger.info(f"Average Perplexity: {avg_perplexity:.4f}")
    logger.info("=" * 60)
    
    # Per-protein accuracy distribution
    if per_protein_results:
        accuracies = [r['accuracy'] for r in per_protein_results]
        logger.info(f"Per-protein accuracy - Min: {min(accuracies):.4f}, Max: {max(accuracies):.4f}, "
                   f"Median: {np.median(accuracies):.4f}")
    
    # Save results
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Detailed results saved to {results_file}")
    
    # Save CSV summary for easy analysis
    csv_file = output_dir / "test_results_per_protein.csv"
    with open(csv_file, 'w') as f:
        f.write("name,length,loss,accuracy,perplexity,correct_residues\n")
        for r in per_protein_results:
            f.write(f"{r['name']},{r['length']},{r['loss']:.6f},{r['accuracy']:.6f},"
                   f"{r['perplexity']:.6f},{r['correct_residues']}\n")
    logger.info(f"CSV summary saved to {csv_file}")
    
    return results_summary



# ==============================================================================
# MAIN
# ==============================================================================

def load_pretrained_weights(model, checkpoint_path, strict=False):
    """Load pretrained ProteinMPNN weights"""
    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"  {k}")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"  {k}")

    print("Weights loaded successfully!")
    return checkpoint

def main():
    parser = argparse.ArgumentParser(description="Fine-tune or Test ProteinMPNN with LoRA, Layer Freezing, and EWC")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Training data directory (required unless --test is used)")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Validation data directory")
    parser.add_argument("--test_data_dir", type=str, default=None,
                        help="Test data directory for evaluation")
    parser.add_argument("--test", action="store_true",
                        help="Run evaluation on test set instead of training")
    parser.add_argument("--fixed_positions_json", type=str, default=None,
                        help="JSON file specifying fixed chains/positions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pretrained checkpoint path (required for --test)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for results/checkpoints")

    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--k_neighbors", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--backbone_noise", type=float, default=0.0,
                        help="Noise added to backbone coordinates (disabled during test)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA (Low-Rank Adaptation)")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Rank for LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA scaling parameter (alpha)")
    parser.add_argument("--lora_target_modules", type=str, default="W1,W2,W3,W11,W12,W13,W_in,W_out",
                        help="Comma-separated list of module names to apply LoRA to")

    # Layer freezing arguments
    parser.add_argument("--freeze_encoder_layers", type=int, default=0,
                        help="Number of encoder layers to freeze from the start")
    parser.add_argument("--freeze_decoder_layers", type=int, default=0,
                        help="Number of decoder layers to freeze from the start")

    # EWC arguments
    parser.add_argument("--use_ewc", action="store_true",
                        help="Enable Elastic Weight Consolidation")
    parser.add_argument("--ewc_lambda", type=float, default=1000.0,
                        help="EWC regularization strength")
    parser.add_argument("--ewc_sample_size", type=int, default=200,
                        help="Number of samples for Fisher Information Matrix computation")

    # Other arguments
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")

    args = parser.parse_args()

    # Validate arguments
    if args.test:
        if not args.test_data_dir:
            parser.error("--test_data_dir is required when using --test")
        if not args.checkpoint:
            parser.error("--checkpoint is required when using --test")
    else:
        if not args.data_dir:
            parser.error("--data_dir is required for training")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load fixed positions config
    fixed_chains, fixed_positions = {}, {}
    if args.fixed_positions_json:
        with open(args.fixed_positions_json) as f:
            config = json.load(f)
            for name, settings in config.items():
                if 'fixed_chains' in settings:
                    fixed_chains[name] = settings['fixed_chains']
                if 'fixed_positions' in settings:
                    fixed_positions[name] = settings['fixed_positions']

    # Model
    model = ProteinMPNN(
        num_letters=21,
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.k_neighbors,
        augment_eps=0.0 if args.test else args.backbone_noise,
        dropout=args.dropout if not args.test else 0.0
    )

    # Load checkpoint
    if args.checkpoint:
        load_pretrained_weights(model, args.checkpoint, strict=False)

    # Apply LoRA if enabled
    if args.use_lora and not args.test:
        print("\n" + "="*60)
        print("APPLYING LORA")
        print("="*60)
        target_modules = args.lora_target_modules.split(',')
        add_lora_to_linear(model, rank=args.lora_rank, alpha=args.lora_alpha, 
                          target_modules=target_modules)
        lora_params, total_params = count_lora_parameters(model)
        print(f"LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}% of total)")
        print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
        print("="*60 + "\n")

    # Freeze layers if specified
    if (args.freeze_encoder_layers > 0 or args.freeze_decoder_layers > 0) and not args.test:
        print("\n" + "="*60)
        print("FREEZING LAYERS")
        print("="*60)
        frozen_params, layer_params = freeze_layers(
            model, 
            freeze_encoder_layers=args.freeze_encoder_layers,
            freeze_decoder_layers=args.freeze_decoder_layers
        )
        print(f"Frozen encoder layers: {args.freeze_encoder_layers}/{args.num_encoder_layers}")
        print(f"Frozen decoder layers: {args.freeze_decoder_layers}/{args.num_decoder_layers}")
        print(f"Frozen parameters: {frozen_params:,}")
        print("="*60 + "\n")

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/num_params*100:.2f}%)")

    # Config for data loading
    config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'k_neighbors': args.k_neighbors,
    }

    if args.test:
        # ===================== TEST MODE =====================
        print("\n" + "="*60)
        print("RUNNING IN TEST MODE")
        print("="*60 + "\n")

        test_dataset = ProteinDataset(
            args.test_data_dir,
            fixed_chains_dict=fixed_chains,
            fixed_positions_dict=fixed_positions
        )

        results = run_test(model, test_dataset, config, args.output_dir, device)

        print("\n" + "="*60)
        print("TEST COMPLETE")
        print(f"Results saved to: {args.output_dir}")
        print("="*60)

    else:
        # ===================== TRAINING MODE =====================
        print("\n" + "="*60)
        print("RUNNING IN TRAINING MODE")
        print("="*60 + "\n")

        # Datasets
        train_dataset = ProteinDataset(
            args.data_dir,
            fixed_chains_dict=fixed_chains,
            fixed_positions_dict=fixed_positions
        )

        val_dataset = None
        if args.val_data_dir:
            val_dataset = ProteinDataset(
                args.val_data_dir,
                fixed_chains_dict=fixed_chains,
                fixed_positions_dict=fixed_positions
            )

        # Initialize EWC if enabled
        ewc = None
        if args.use_ewc:
            print("\n" + "="*60)
            print("INITIALIZING EWC")
            print("="*60)
            ewc = EWC(model, train_dataset, device, sample_size=args.ewc_sample_size)
            print(f"EWC lambda: {args.ewc_lambda}")
            print("="*60 + "\n")

        # Training config
        config.update({
            'num_epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'grad_clip': args.grad_clip,
            'label_smoothing': args.label_smoothing,
            'log_interval': args.log_interval,
            'eval_interval': args.eval_interval,
            'save_interval': args.save_interval,
            'use_amp': True,
            'use_ewc': args.use_ewc,
            'ewc_lambda': args.ewc_lambda
        })

        trainer = Trainer(model, train_dataset, val_dataset, config, args.output_dir, device, ewc=ewc)
        trainer.train(resume_from=args.resume)

if __name__ == "__main__":
    main()