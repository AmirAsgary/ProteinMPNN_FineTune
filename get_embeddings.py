#!/usr/bin/env python3
"""
Extract encoder embeddings from ProteinMPNN and save incrementally to HDF5.

Usage:
    python extract_embeddings.py --checkpoint model.pt --data_dir ./pdbs --output embeddings.h5
    python extract_embeddings.py --checkpoint model.pt --data_dir ./pdbs --output embeddings.h5 --pool
"""

import argparse
import h5py
import numpy as np
import torch
from finetune_proteinmpnn import (
    ProteinMPNN, ProteinDataset, collate_fn, gather_nodes
)
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_and_save(model, dataloader, device, output_path, pool=False):
    """Extract encoder embeddings and save incrementally to HDF5."""
    model.eval()
    
    with h5py.File(output_path, 'w') as f:
        # Create extensible datasets for pooled embeddings
        if pool:
            hidden_dim = model.hidden_dim
            f.create_dataset('h_V', shape=(0, hidden_dim), maxshape=(None, hidden_dim), dtype='float32')
            f.create_dataset('h_E', shape=(0, hidden_dim), maxshape=(None, hidden_dim), dtype='float32')
            f.create_dataset('lengths', shape=(0,), maxshape=(None,), dtype='int32')
        
        names_list = []
        idx = 0
        
        for batch in dataloader:
            X = batch['X'].to(device)
            mask = batch['mask'].to(device)
            residue_idx = batch['residue_idx'].to(device)
            chain_encoding = batch['chain_encoding_all'].to(device)
            names = batch['names']
            
            # Get features and run encoder
            E, E_idx = model.features(X, mask, residue_idx, chain_encoding)
            h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
            h_E = model.W_e(E)
            
            mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
            mask_attend = mask.unsqueeze(-1) * mask_attend
            
            for layer in model.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            
            # Process and save each protein immediately
            for i, name in enumerate(names):
                length = int(mask[i].sum().item())
                h_V_i = h_V[i, :length].cpu().numpy()
                h_E_i = h_E[i, :length].cpu().numpy()
                
                if pool:
                    h_V_pooled = h_V_i.mean(axis=0)
                    h_E_pooled = h_E_i.mean(axis=(0, 1))
                    
                    # Extend datasets
                    f['h_V'].resize(idx + 1, axis=0)
                    f['h_E'].resize(idx + 1, axis=0)
                    f['lengths'].resize(idx + 1, axis=0)
                    
                    f['h_V'][idx] = h_V_pooled
                    f['h_E'][idx] = h_E_pooled
                    f['lengths'][idx] = length
                else:
                    # Save variable-length arrays as separate datasets
                    f.create_dataset(f'{name}/h_V', data=h_V_i, compression='gzip')
                    f.create_dataset(f'{name}/h_E', data=h_E_i, compression='gzip')
                    f[f'{name}'].attrs['length'] = length
                
                names_list.append(name)
                idx += 1
                
                if idx % 100 == 0:
                    print(f"Processed {idx} proteins...")
        
        # Save names
        f.create_dataset('names', data=np.array(names_list, dtype='S'))
        
    print(f"Saved {idx} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract ProteinMPNN encoder embeddings")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="embeddings.h5")
    parser.add_argument("--pool", action="store_true", help="Pool to fixed-size vectors")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--k_neighbors", type=int, default=48)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = ProteinMPNN(
        hidden_dim=config.get('hidden_dim', args.hidden_dim),
        num_encoder_layers=config.get('num_encoder_layers', args.num_encoder_layers),
        num_decoder_layers=config.get('num_decoder_layers', args.num_decoder_layers),
        k_neighbors=checkpoint.get('num_edges', config.get('k_neighbors', args.k_neighbors)),
        dropout=0.0, augment_eps=0.0
    )
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"Loaded model from {args.checkpoint}")
    
    # Load data
    dataset = ProteinDataset(args.data_dir, verbose=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, pin_memory=True)
    
    # Extract and save incrementally
    print(f"Extracting embeddings (pool={args.pool})...")
    extract_and_save(model, loader, device, args.output, pool=args.pool)


if __name__ == "__main__":
    main()
