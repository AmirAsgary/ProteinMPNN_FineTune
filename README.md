# ProteinMPNN Fine-tuning Framework

A comprehensive framework for fine-tuning ProteinMPNN on your own protein structure data with full control over training, fixed positions/chains, and monitoring.

## Features

- **Custom Data Support**: Load PDB files or JSON/JSONL formatted data
- **Fixed Positions/Chains**: Specify which residues or chains should remain fixed during training and inference
- **Training Monitoring**: TensorBoard and Weights & Biases integration
- **Flexible Configuration**: YAML config files or command-line arguments
- **Mixed Precision Training**: Faster training with automatic mixed precision (AMP)
- **Checkpointing**: Save and resume training, automatic best model saving
- **Inference Script**: Sample sequences with various temperatures and biases

## Installation

### Prerequisites

```bash
# Create conda environment
conda create -n proteinmpnn python=3.10
conda activate proteinmpnn

# Install PyTorch (adjust for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio
```

### Required Packages

```bash
pip install numpy pyyaml tqdm
```

### Optional Packages

```bash
# For monitoring
pip install tensorboard wandb

# For PDB parsing (recommended)
pip install biopython
```

### Download Pretrained Weights

```bash
# Create directory for pretrained weights
mkdir -p pretrained

# Download from ProteinMPNN repository
# Option 1: v_48_020 (recommended, 0.20Å noise)
wget https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt -O pretrained/v_48_020.pt

# Option 2: v_48_010 (0.10Å noise)
wget https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_010.pt -O pretrained/v_48_010.pt

# You can also use soluble model weights for soluble proteins:
wget https://github.com/dauparas/ProteinMPNN/raw/main/soluble_model_weights/v_48_020.pt -O pretrained/soluble_v_48_020.pt
```

## Quick Start

### 1. Prepare Your Data

Organize your PDB files in a directory:

```
my_data/
├── protein1.pdb
├── protein2.pdb
├── protein3.pdb
└── ...
```

Run the data preparation script:

```bash
python prepare_data.py \
    --pdb_dir ./my_data \
    --output_dir ./prepared_data \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

This creates:
```
prepared_data/
├── train/
│   └── data.jsonl
├── val/
│   └── data.jsonl
└── test/
    └── data.jsonl
```

### 2. Fine-tune the Model

Basic fine-tuning:

```bash
python finetune_proteinmpnn.py \
    --data_dir ./prepared_data/train \
    --val_data_dir ./prepared_data/val \
    --checkpoint ./pretrained/v_48_020.pt \
    --output_dir ./outputs/my_finetuned_model \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 8
```

Using a config file:

```bash
python finetune_proteinmpnn.py --config config.yaml
```

### 3. Sample Sequences

```bash
python sample_sequences.py \
    --checkpoint ./outputs/my_finetuned_model/checkpoints/checkpoint_best.pt \
    --pdb_path ./my_protein.pdb \
    --output_dir ./designed_sequences \
    --num_samples 100 \
    --temperature 0.1
```

## Fixed Positions and Chains

### During Training

Create a `fixed_positions.json` file:

```json
{
    "protein1": {
        "fixed_chains": ["A", "B"],
        "fixed_positions": {
            "C": [1, 2, 3, 10, 11, 12]
        }
    },
    "protein2": {
        "fixed_chains": ["A"],
        "fixed_positions": {
            "B": [50, 51, 52]
        }
    }
}
```

Then run:

```bash
python finetune_proteinmpnn.py \
    --data_dir ./prepared_data/train \
    --fixed_positions_json ./fixed_positions.json \
    --checkpoint ./pretrained/v_48_020.pt \
    --output_dir ./outputs/my_model
```

### During Inference

Fix entire chains:

```bash
python sample_sequences.py \
    --checkpoint ./outputs/my_model/checkpoints/checkpoint_best.pt \
    --pdb_path ./my_protein.pdb \
    --fixed_chains A,B \
    --num_samples 100
```

Fix specific positions:

```bash
python sample_sequences.py \
    --checkpoint ./outputs/my_model/checkpoints/checkpoint_best.pt \
    --pdb_path ./my_protein.pdb \
    --fixed_positions "A:1,2,3,10-15;B:5,6,7" \
    --num_samples 100
```

Design only specific chains:

```bash
python sample_sequences.py \
    --checkpoint ./outputs/my_model/checkpoints/checkpoint_best.pt \
    --pdb_path ./my_protein.pdb \
    --design_only C \
    --num_samples 100
```

## Configuration Options

### Training Configuration (config.yaml)

```yaml
# Data
data_dir: "./data/train"
val_data_dir: "./data/val"
fixed_positions_json: null

# Model Architecture
hidden_dim: 128
num_encoder_layers: 3
num_decoder_layers: 3
k_neighbors: 48
dropout: 0.1
backbone_noise: 0.1

# Pretrained checkpoint
checkpoint: "./pretrained/v_48_020.pt"

# Training
output_dir: "./outputs/my_model"
epochs: 100
batch_size: 8
lr: 1e-4
weight_decay: 0.01
warmup_steps: 1000
grad_clip: 1.0
scheduler: "cosine"  # cosine, onecycle, plateau, none

# Mixed precision
use_amp: true

# Monitoring
use_wandb: false
wandb_project: "proteinmpnn-finetune"
log_interval: 10
eval_interval: 1
save_interval: 5

# Other
num_workers: 4
seed: 42
```

### Key Hyperparameters

| Parameter | Recommended Range | Description |
|-----------|------------------|-------------|
| `lr` | 1e-5 to 1e-4 | Lower for fine-tuning, higher for training from scratch |
| `backbone_noise` | 0.02 to 0.30 | Gaussian noise added to backbone coordinates |
| `dropout` | 0.1 | Dropout rate |
| `batch_size` | 4-32 | Depends on GPU memory and protein length |
| `k_neighbors` | 48 | Number of neighbors in graph (keep same as pretrained) |

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./outputs/my_model/tensorboard
```

Then open http://localhost:6006 in your browser.

### Weights & Biases

```bash
python finetune_proteinmpnn.py \
    --config config.yaml \
    --use_wandb \
    --wandb_project my_project
```

## Output Files

After training:
```
outputs/my_model/
├── checkpoints/
│   ├── checkpoint_best.pt      # Best validation loss
│   ├── checkpoint_latest.pt    # Most recent
│   └── checkpoint_epoch_*.pt   # Periodic saves
├── logs/
│   └── training_*.log          # Training logs
└── tensorboard/
    └── events.*                # TensorBoard logs
```

After sampling:
```
designed_sequences/
├── my_protein_designed.fasta   # FASTA with all sequences
└── my_protein_results.json     # Detailed results
```

## Advanced Usage

### Resume Training

```bash
python finetune_proteinmpnn.py \
    --config config.yaml \
    --resume ./outputs/my_model/checkpoints/checkpoint_latest.pt
```

### Amino Acid Biases During Sampling

Reduce cysteine and increase hydrophobics:

```bash
python sample_sequences.py \
    --checkpoint ./outputs/my_model/checkpoints/checkpoint_best.pt \
    --pdb_path ./my_protein.pdb \
    --bias_AA "C:-5.0,W:1.0,F:1.0,Y:1.0" \
    --num_samples 100
```

### Multi-temperature Sampling

```bash
python sample_sequences.py \
    --checkpoint ./outputs/my_model/checkpoints/checkpoint_best.pt \
    --pdb_path ./my_protein.pdb \
    --temperature "0.1 0.2 0.3" \
    --num_samples 100
```

## Tips for Fine-tuning

1. **Start with lower learning rate**: Use 1e-5 to 1e-4 for fine-tuning
2. **Match backbone noise**: Use the same noise level as the pretrained model (0.20Å for v_48_020)
3. **Monitor validation loss**: Watch for overfitting, especially with small datasets
4. **Data quality**: Ensure PDB files have good resolution and complete backbone atoms
5. **Batch size**: Reduce if running out of GPU memory
6. **Fixed positions**: Training with fixed positions teaches the model to focus on designable regions

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Enable `use_amp: true` (mixed precision)
- Reduce `max_length` for proteins

### Poor Performance
- Check that pretrained weights loaded correctly
- Ensure data format is correct
- Try different learning rates
- Increase training epochs

### NaN Loss
- Reduce learning rate
- Check for corrupted PDB files
- Enable gradient clipping (default: 1.0)

## Citation

If you use this code, please cite the original ProteinMPNN paper:

```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

## License

This fine-tuning framework is provided under MIT License. The original ProteinMPNN code is also under MIT License.
