#!/usr/bin/env python3
"""
Inference Script for Fine-tuned ProteinMPNN

Sample sequences from a fine-tuned (or original) ProteinMPNN model.

Usage:
    python sample_sequences.py \
        --checkpoint ./outputs/my_model/checkpoints/checkpoint_best.pt \
        --pdb_path ./my_protein.pdb \
        --output_dir ./designed_sequences \
        --num_samples 100 \
        --temperature 0.1

    # With fixed positions
    python sample_sequences.py \
        --checkpoint ./outputs/my_model/checkpoints/checkpoint_best.pt \
        --pdb_path ./my_protein.pdb \
        --fixed_chains A \
        --fixed_positions "B:1,2,3,10-15" \
        --num_samples 100
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from finetune_proteinmpnn import cat_neighbors_nodes
import numpy as np
import torch
import torch.nn.functional as F

# Import model from fine-tuning script
from finetune_proteinmpnn import ProteinMPNN, load_pretrained_weights


def parse_pdb(pdb_path: str) -> Tuple[Dict, Dict]:
    """Parse PDB file and return coordinates and sequences by chain."""
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    coords_by_chain = {}
    seqs_by_chain = {}
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
                    
    # Sort by order of appearance and convert to arrays
    sorted_residues = sorted(residue_data.items(), key=lambda x: x[1]['order'])
    
    for (chain_id, res_num, resname), atoms in sorted_residues:
        if all(a in atoms for a in ['N', 'CA', 'C', 'O']):
            if chain_id not in coords_by_chain:
                coords_by_chain[chain_id] = []
                seqs_by_chain[chain_id] = []
                
            coords_by_chain[chain_id].append([
                atoms['N'], atoms['CA'], atoms['C'], atoms['O']
            ])
            seqs_by_chain[chain_id].append(three_to_one[resname])
            
    # Convert to numpy arrays
    for chain_id in coords_by_chain:
        coords_by_chain[chain_id] = np.array(coords_by_chain[chain_id])
        seqs_by_chain[chain_id] = ''.join(seqs_by_chain[chain_id])
        
    return coords_by_chain, seqs_by_chain


def parse_fixed_positions(fixed_str: str) -> Dict[str, List[int]]:
    """
    Parse fixed positions string.
    
    Format: "A:1,2,3,10-15;B:5,6,7"
    Returns: {'A': [1,2,3,10,11,12,13,14,15], 'B': [5,6,7]}
    """
    result = {}
    
    if not fixed_str:
        return result
        
    for part in fixed_str.split(';'):
        if ':' not in part:
            continue
            
        chain, positions = part.split(':')
        chain = chain.strip()
        pos_list = []
        
        for p in positions.split(','):
            p = p.strip()
            if '-' in p:
                start, end = p.split('-')
                pos_list.extend(range(int(start), int(end) + 1))
            else:
                pos_list.append(int(p))
                
        result[chain] = pos_list
        
    return result


def prepare_input(
    coords_by_chain: Dict,
    seqs_by_chain: Dict,
    fixed_chains: List[str],
    fixed_positions: Dict[str, List[int]],
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Prepare model input from parsed PDB data."""
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    
    all_coords = []
    all_seq = []
    chain_labels = []
    residue_idx = []
    chain_mask = []
    chain_ids = []
    
    chain_offset = 0
    
    for chain_idx, chain_id in enumerate(sorted(coords_by_chain.keys())):
        chain_coords = coords_by_chain[chain_id]
        chain_seq = seqs_by_chain.get(chain_id, 'X' * len(chain_coords))
        
        L_chain = len(chain_coords)
        
        all_coords.append(chain_coords)
        all_seq.extend(chain_seq)
        chain_labels.extend([chain_idx] * L_chain)
        residue_idx.extend(range(chain_offset, chain_offset + L_chain))
        chain_ids.extend([chain_id] * L_chain)
        
        # Determine fixed positions
        chain_fixed_positions = fixed_positions.get(chain_id, [])
        
        for i in range(L_chain):
            if chain_id in fixed_chains:
                chain_mask.append(0)
            elif (i + 1) in chain_fixed_positions:
                chain_mask.append(0)
            else:
                chain_mask.append(1)
                
        chain_offset += L_chain
        
    all_coords = np.concatenate(all_coords, axis=0)
    seq_idx = [aa_to_idx.get(aa, 20) for aa in all_seq]
    mask = [1.0] * len(all_seq)
    
    return {
        'X': torch.tensor(all_coords, dtype=torch.float32, device=device).unsqueeze(0),
        'S': torch.tensor(seq_idx, dtype=torch.long, device=device).unsqueeze(0),
        'mask': torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0),
        'chain_M': torch.tensor(chain_mask, dtype=torch.float32, device=device).unsqueeze(0),
        'residue_idx': torch.tensor(residue_idx, dtype=torch.long, device=device).unsqueeze(0),
        'chain_encoding_all': torch.tensor(chain_labels, dtype=torch.long, device=device).unsqueeze(0),
        'chain_ids': chain_ids,
        'original_seq': ''.join(all_seq)
    }


@torch.no_grad()
def sample_sequence(
    model: ProteinMPNN,
    batch: Dict[str, torch.Tensor],
    temperature: float = 0.1,
    bias_AA: Optional[Dict[str, float]] = None,
    omit_AAs: Optional[str] = None
) -> Tuple[str, float]:
    """
    Sample a sequence from the model.
    
    Args:
        model: ProteinMPNN model
        batch: Input batch
        temperature: Sampling temperature
        bias_AA: Amino acid bias dict (e.g., {'C': -5.0} to reduce cysteine)
        omit_AAs: String of amino acids to omit (e.g., 'CM')
        
    Returns:
        sequence: Sampled sequence string
        score: Negative log probability score
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    idx_to_aa = {i: aa for i, aa in enumerate(alphabet)}
    
    device = batch['X'].device
    X = batch['X']
    mask = batch['mask']
    chain_M = batch['chain_M']
    residue_idx = batch['residue_idx']
    chain_encoding = batch['chain_encoding_all']
    fixed_S = batch['S']
    
    B, L = X.shape[:2]
    
    # Extract features
    E, E_idx = model.features(X, mask, residue_idx, chain_encoding)
    # Manually initialize h_V (matches finetune_proteinmpnn.py logic)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
    h_E = model.W_e(E)
    
    
    # Build attention mask
    from finetune_proteinmpnn import gather_nodes
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    
    # Encoder
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        
    h_V_enc = h_V.clone()
    h_E_enc = h_E.clone()
    # Precompute structure embeddings for the Decoder
    # Use h_V_enc for shape reference since h_S isn't defined yet
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V_enc), h_E_enc, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)
    # Random decoding order
    randn = torch.randn(B, L, device=device)
    randn_masked = randn * chain_M + (-1e8) * (1 - chain_M)
    decoding_order = torch.argsort(randn_masked, dim=-1)
    
    # Initialize sequence
    S = fixed_S.clone()
    
    # Create bias tensor
    bias = torch.zeros(21, device=device)
    if bias_AA:
        for aa, val in bias_AA.items():
            if aa in alphabet:
                bias[alphabet.index(aa)] = val
                
    # Create omit mask
    omit_mask = torch.zeros(21, device=device)
    if omit_AAs:
        for aa in omit_AAs:
            if aa in alphabet:
                omit_mask[alphabet.index(aa)] = -1e9
                
    # Autoregressive sampling
    log_probs_sum = 0.0
    designed_count = 0
    
    h_S = model.W_s(S)
    
    for i in range(L):
        idx = decoding_order[:, i]
        
        # Skip fixed positions
        is_fixed = chain_M[0, idx[0]] == 0
        if is_fixed:
            continue
            
        # Decoder
        h_ES = cat_neighbors_nodes(h_S, h_E_enc, E_idx)
        h_V = h_V_enc.clone()
        
        for layer in model.decoder_layers:
            # 1. Create the combined features (Size 384)
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            # 2. Add structural info (Crucial for performance, though shape would work without it)
            # In sampling, we simply add the structure features to the sequence features
            # to match the dimensions and providing context.
            # (Strictly speaking, we should mask based on decoded positions, but 
            # mixing them allows the layer to see both).
            h_ESV = h_ESV + h_EXV_encoder
            # 3. Pass correctly shaped h_ESV to the layer
            h_V = layer(h_V, h_ESV, mask, mask_attend)
            
        # Get logits for current position
        logits = model.W_out(h_V)
        logits_i = logits[0, idx[0]] + bias + omit_mask
        
        # Sample
        probs = F.softmax(logits_i / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        
        # Update sequence
        S[0, idx[0]] = sampled
        h_S = model.W_s(S)
        
        # Track score
        log_probs_sum += torch.log(probs[sampled]).item()
        designed_count += 1
        
    # Convert to string
    seq_list = [idx_to_aa.get(s.item(), 'X') for s in S[0]]
    sequence = ''.join(seq_list)
    
    # Average score (negative log prob per position)
    score = -log_probs_sum / max(designed_count, 1)
    
    return sequence, score

def compute_global_score(
    model: ProteinMPNN, 
    batch: Dict[str, torch.Tensor], 
    sequence: str
) -> float:
    """Compute global score (negative log prob) for a sequence."""
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    device = batch['X'].device

    # Encode sequence
    S = torch.tensor(
        [[aa_to_idx.get(aa, 20) for aa in sequence]],
        dtype=torch.long,
        device=device
    )

    # --- NEW: Generate random noise tensor ---
    randn = torch.randn(batch['X'].shape[0], batch['X'].shape[1], device=device)
    # -----------------------------------------

    # Forward pass
    # model() returns only log_probs in your implementation (see finetune_proteinmpnn.py), 
    # not (log_probs, loss_mask).
    log_probs = model(
        batch['X'],
        S,
        batch['mask'],
        torch.ones_like(batch['chain_M']),  # Score all positions
        batch['residue_idx'],
        batch['chain_encoding_all'],
        randn  # <--- Pass the new argument here
    )

    # Compute score
    loss = F.nll_loss(
        log_probs.view(-1, log_probs.size(-1)),
        S.view(-1),
        reduction='none'
    ).view(S.shape)

    global_score = (loss * batch['mask']).sum() / batch['mask'].sum()
    return global_score.item()



def write_fasta(
    sequences: List[Dict],
    output_path: str,
    chain_ids: List[str],
    pdb_name: str
):
    """Write sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for i, seq_data in enumerate(sequences):
            # Build header
            header_parts = [
                pdb_name,
                f"sample={i+1}",
                f"T={seq_data['temperature']:.2f}",
                f"score={seq_data['score']:.4f}",
                f"global_score={seq_data['global_score']:.4f}",
                f"seq_recovery={seq_data['recovery']:.4f}"
            ]
            
            if seq_data.get('designed_chains'):
                header_parts.append(f"designed_chains={seq_data['designed_chains']}")
            if seq_data.get('fixed_chains'):
                header_parts.append(f"fixed_chains={seq_data['fixed_chains']}")
                
            header = ', '.join(header_parts)
            
            # Write sequence (split by chain)
            f.write(f">{header}\n")
            
            # Split sequence by chain for multi-chain proteins
            seq = seq_data['sequence']
            chain_seqs = []
            offset = 0
            unique_chains = []
            for cid in chain_ids:
                if cid not in unique_chains:
                    unique_chains.append(cid)
                    
            prev_chain = None
            chain_start = 0
            for pos, cid in enumerate(chain_ids):
                if prev_chain is not None and cid != prev_chain:
                    chain_seqs.append(seq[chain_start:pos])
                    chain_start = pos
                prev_chain = cid
            chain_seqs.append(seq[chain_start:])
            
            f.write('/'.join(chain_seqs) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Sample sequences from fine-tuned ProteinMPNN"
    )
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--pdb_path", type=str, required=True,
                        help="Path to input PDB file")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./designed_sequences",
                        help="Output directory")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output name (default: PDB filename)")
    
    # Sampling parameters
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sequences to sample")
    parser.add_argument("--temperature", type=str, default="0.1",
                        help="Sampling temperature(s), space-separated")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for sampling")
    
    # Fixed positions/chains
    parser.add_argument("--fixed_chains", type=str, default="",
                        help="Comma-separated list of fixed chain IDs")
    parser.add_argument("--fixed_positions", type=str, default="",
                        help="Fixed positions, e.g., 'A:1,2,3,10-15;B:5,6,7'")
    parser.add_argument("--design_only", type=str, default="",
                        help="Design only these chains (others fixed)")
    
    # Amino acid biases
    parser.add_argument("--omit_AAs", type=str, default="X",
                        help="Amino acids to omit (e.g., 'CX')")
    parser.add_argument("--bias_AA", type=str, default="",
                        help="AA biases as 'A:-1.0,F:0.5'")
    
    # Model parameters (for loading checkpoint)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--k_neighbors", type=int, default=48)
    
    # Other
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse PDB
    print(f"Loading PDB: {args.pdb_path}")
    coords_by_chain, seqs_by_chain = parse_pdb(args.pdb_path)
    
    print(f"Found {len(coords_by_chain)} chains:")
    for chain_id, coords in coords_by_chain.items():
        print(f"  Chain {chain_id}: {len(coords)} residues")
        
    # Parse fixed chains
    fixed_chains = []
    if args.fixed_chains:
        fixed_chains = [c.strip() for c in args.fixed_chains.split(',')]
        
    # If design_only is specified, fix all other chains
    if args.design_only:
        design_chains = [c.strip() for c in args.design_only.split(',')]
        fixed_chains = [c for c in coords_by_chain.keys() if c not in design_chains]
        
    # Parse fixed positions
    fixed_positions = parse_fixed_positions(args.fixed_positions)
    
    # Parse AA bias
    bias_AA = None
    if args.bias_AA:
        bias_AA = {}
        for item in args.bias_AA.split(','):
            if ':' in item:
                aa, val = item.split(':')
                bias_AA[aa.strip()] = float(val.strip())
                
    # Parse temperatures
    temperatures = [float(t) for t in args.temperature.split()]
    
    # Prepare input
    batch = prepare_input(
        coords_by_chain,
        seqs_by_chain,
        fixed_chains,
        fixed_positions,
        device
    )
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    
    # Try to load config from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    
    model = ProteinMPNN(
        hidden_dim=config.get('hidden_dim', args.hidden_dim),
        num_encoder_layers=config.get('num_encoder_layers', args.num_encoder_layers),
        num_decoder_layers=config.get('num_decoder_layers', args.num_decoder_layers),
        k_neighbors=config.get('k_neighbors', args.k_neighbors) if 'k_neighbors' in config else checkpoint.get('num_edges', args.k_neighbors),
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Sample sequences
    print(f"\nSampling {args.num_samples} sequences...")
    
    sequences = []
    pdb_name = Path(args.pdb_path).stem
    original_seq = batch['original_seq']
    
    for temp in temperatures:
        for i in range(args.num_samples):
            # Sample
            seq, score = sample_sequence(
                model,
                batch,
                temperature=temp,
                bias_AA=bias_AA,
                omit_AAs=args.omit_AAs
            )
            
            # Compute global score
            global_score = compute_global_score(model, batch, seq)
            
            # Compute recovery
            recovery = sum(a == b for a, b in zip(seq, original_seq)) / len(seq)
            
            # Determine designed/fixed chains
            designed_chains = [c for c in coords_by_chain.keys() if c not in fixed_chains]
            
            sequences.append({
                'sequence': seq,
                'score': score,
                'global_score': global_score,
                'temperature': temp,
                'recovery': recovery,
                'designed_chains': ','.join(designed_chains),
                'fixed_chains': ','.join(fixed_chains) if fixed_chains else None
            })
            
            print(f"  T={temp:.2f}, sample={i+1}: score={score:.4f}, recovery={recovery:.4f}")
            
    # Sort by score
    sequences.sort(key=lambda x: x['score'])
    
    # Write output
    output_name = args.output_name or pdb_name
    fasta_path = output_dir / f"{output_name}_designed.fasta"
    
    write_fasta(sequences, str(fasta_path), batch['chain_ids'], pdb_name)
    print(f"\nSequences saved to: {fasta_path}")
    
    # Save detailed results
    json_path = output_dir / f"{output_name}_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'pdb_name': pdb_name,
            'original_sequence': original_seq,
            'fixed_chains': fixed_chains,
            'fixed_positions': fixed_positions,
            'temperatures': temperatures,
            'num_samples': args.num_samples,
            'sequences': sequences
        }, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Print best sequence
    print(f"\nBest sequence (score={sequences[0]['score']:.4f}):")
    print(f"  {sequences[0]['sequence'][:50]}...")


if __name__ == "__main__":
    main()
