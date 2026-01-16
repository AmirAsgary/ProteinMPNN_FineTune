#!/usr/bin/env python3
"""
Data Preparation Script for ProteinMPNN Fine-tuning

This script helps prepare your PDB data for fine-tuning:
1. Parse PDB files into JSON format
2. Create train/val splits
3. Generate fixed positions configuration

Usage:
    # Basic preparation
    python prepare_data.py --pdb_dir ./raw_pdbs --output_dir ./data
    
    # With fixed chains
    python prepare_data.py --pdb_dir ./raw_pdbs --output_dir ./data \
        --fixed_chains "A,B" 
    
    # With specific fixed positions from a file
    python prepare_data.py --pdb_dir ./raw_pdbs --output_dir ./data \
        --fixed_positions_file fixed_residues.txt
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def parse_pdb(pdb_path: str) -> Optional[Dict]:
    """
    Parse a PDB file and extract backbone coordinates and sequence.
    Returns a dictionary in ProteinMPNN-compatible format.
    """
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    coords_by_chain = {}
    seqs_by_chain = {}
    residue_data = {}
    pdb_name = Path(pdb_path).stem

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
                    residue_data[key] = {}
                
                if atom_name in ['N', 'CA', 'C', 'O']:
                    residue_data[key][atom_name] = [x, y, z]

    # Convert to lists
    for (chain_id, res_num, resname), atoms in residue_data.items():
        if all(a in atoms for a in ['N', 'CA', 'C', 'O']):
            if chain_id not in coords_by_chain:
                coords_by_chain[chain_id] = []
                seqs_by_chain[chain_id] = []

            coords_by_chain[chain_id].append([
                atoms['N'], atoms['CA'], atoms['C'], atoms['O']
            ])
            seqs_by_chain[chain_id].append(three_to_one[resname])

    if not coords_by_chain:
        return None

    # Build output dictionary (Corrected Format)
    result = {'name': pdb_name}
    for chain_id in sorted(coords_by_chain.keys()):
        result[f'seq_chain_{chain_id}'] = ''.join(seqs_by_chain[chain_id])
        
        # Create the dictionary structure expected by the dataloader
        chain_coords = coords_by_chain[chain_id]
        result[f'coords_chain_{chain_id}'] = {
            f'N_chain_{chain_id}': [res[0] for res in chain_coords],
            f'CA_chain_{chain_id}': [res[1] for res in chain_coords],
            f'C_chain_{chain_id}': [res[2] for res in chain_coords],
            f'O_chain_{chain_id}': [res[3] for res in chain_coords]
        }

    return result



def parse_fixed_positions_file(filepath: str) -> Dict:
    """
    Parse a fixed positions file.
    
    Expected format (one entry per line):
    pdb_name chain position1,position2,position3,...
    
    Example:
    1abc A 1,2,3,10,11,12
    1abc B 5,6,7
    2def C 100,101,102
    """
    fixed_positions = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                pdb_name = parts[0]
                chain = parts[1]
                positions = [int(p) for p in parts[2].split(',')]
                
                if pdb_name not in fixed_positions:
                    fixed_positions[pdb_name] = {}
                fixed_positions[pdb_name][chain] = positions
                
    return fixed_positions


def create_jsonl(pdb_files: List[str], output_path: str, verbose: bool = True) -> List[str]:
    """
    Convert PDB files to JSONL format.
    Returns list of successfully parsed PDB names.
    """
    parsed_names = []
    
    with open(output_path, 'w') as f:
        for pdb_path in pdb_files:
            try:
                data = parse_pdb(pdb_path)
                if data is not None:
                    # REMOVED: The loop that corrupted the dictionary
                    # "for key in data: if key.startswith('coords_')..." 
                    # parse_pdb now returns pure Python lists/dicts that are already JSON safe.
                    
                    f.write(json.dumps(data) + '\n')
                    parsed_names.append(data['name'])
                    if verbose:
                        print(f"Parsed: {pdb_path}")
                else:
                    if verbose:
                        print(f"Skipped (no valid residues): {pdb_path}")
            except Exception as e:
                if verbose:
                    print(f"Error parsing {pdb_path}: {e}")
                    
    return parsed_names



def create_fixed_config(
    pdb_names: List[str],
    fixed_chains: Optional[List[str]] = None,
    fixed_positions: Optional[Dict] = None,
    output_path: str = "fixed_positions.json"
):
    """
    Create fixed positions configuration file.
    
    Args:
        pdb_names: List of PDB names
        fixed_chains: List of chain IDs to fix (applies to all PDBs)
        fixed_positions: Dict mapping pdb_name -> chain -> positions
        output_path: Output JSON path
    """
    config = {}
    
    for pdb_name in pdb_names:
        entry = {}
        
        # Add fixed chains
        if fixed_chains:
            entry['fixed_chains'] = fixed_chains
            
        # Add fixed positions
        if fixed_positions and pdb_name in fixed_positions:
            entry['fixed_positions'] = fixed_positions[pdb_name]
            
        if entry:
            config[pdb_name] = entry
            
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"Saved fixed positions config to {output_path}")


def split_data(
    pdb_files: List[str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Split PDB files into train/val/test sets."""
    random.seed(seed)
    shuffled = pdb_files.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    
    test_files = shuffled[:n_test]
    val_files = shuffled[n_test:n_test + n_val]
    train_files = shuffled[n_test + n_val:]
    
    return train_files, val_files, test_files


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PDB data for ProteinMPNN fine-tuning"
    )
    
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Directory containing PDB files")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Test set ratio")
    parser.add_argument("--fixed_chains", type=str, default=None,
                        help="Comma-separated list of fixed chain IDs (e.g., 'A,B')")
    parser.add_argument("--fixed_positions_file", type=str, default=None,
                        help="File with fixed positions per PDB")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--copy_pdbs", action="store_true",
                        help="Copy PDB files to output directory")
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)
    
    # Find PDB files
    pdb_dir = Path(args.pdb_dir)
    pdb_files = list(pdb_dir.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files")
    
    if not pdb_files:
        print("No PDB files found!")
        return
        
    # Split data
    train_files, val_files, test_files = split_data(
        [str(f) for f in pdb_files],
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Create JSONL files
    print("\nCreating training data...")
    train_names = create_jsonl(
        train_files, 
        str(output_dir / "train" / "data.jsonl")
    )
    
    print("\nCreating validation data...")
    val_names = create_jsonl(
        val_files, 
        str(output_dir / "val" / "data.jsonl")
    )
    
    print("\nCreating test data...")
    test_names = create_jsonl(
        test_files, 
        str(output_dir / "test" / "data.jsonl")
    )
    
    # Parse fixed chains
    fixed_chains = None
    if args.fixed_chains:
        fixed_chains = [c.strip() for c in args.fixed_chains.split(',')]
        
    # Parse fixed positions
    fixed_positions = None
    if args.fixed_positions_file:
        fixed_positions = parse_fixed_positions_file(args.fixed_positions_file)
        
    # Create fixed positions config
    all_names = train_names + val_names + test_names
    if fixed_chains or fixed_positions:
        create_fixed_config(
            all_names,
            fixed_chains=fixed_chains,
            fixed_positions=fixed_positions,
            output_path=str(output_dir / "fixed_positions.json")
        )
        
    # Copy PDB files if requested
    if args.copy_pdbs:
        import shutil
        
        for f in train_files:
            shutil.copy(f, output_dir / "train")
        for f in val_files:
            shutil.copy(f, output_dir / "val")
        for f in test_files:
            shutil.copy(f, output_dir / "test")
            
    # Print summary
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - train/data.jsonl: {len(train_names)} structures")
    print(f"  - val/data.jsonl: {len(val_names)} structures")
    print(f"  - test/data.jsonl: {len(test_names)} structures")
    
    if fixed_chains or fixed_positions:
        print(f"  - fixed_positions.json: configuration file")
        
    print("\nTo start training:")
    print(f"  python finetune_proteinmpnn.py \\")
    print(f"    --data_dir {output_dir}/train \\")
    print(f"    --val_data_dir {output_dir}/val \\")
    if fixed_chains or fixed_positions:
        print(f"    --fixed_positions_json {output_dir}/fixed_positions.json \\")
    print(f"    --checkpoint ./pretrained/v_48_020.pt \\")
    print(f"    --output_dir ./outputs/my_model")


if __name__ == "__main__":
    main()
