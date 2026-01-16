#!/bin/bash
# Example workflow for ProteinMPNN fine-tuning
# This script demonstrates a complete fine-tuning pipeline

set -e  # Exit on error

echo "=============================================="
echo "ProteinMPNN Fine-tuning Example Workflow"
echo "=============================================="

# Configuration
DATA_DIR="./example_data"
OUTPUT_DIR="./example_outputs"
PRETRAINED_WEIGHTS="./pretrained/v_48_020.pt"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "./pretrained"

# Step 1: Download pretrained weights if not present
echo ""
echo "Step 1: Checking pretrained weights..."
if [ ! -f "$PRETRAINED_WEIGHTS" ]; then
    echo "Downloading pretrained weights..."
    wget -q https://github.com/dauparas/ProteinMPNN/raw/main/vanilla_model_weights/v_48_020.pt \
        -O "$PRETRAINED_WEIGHTS"
    echo "Downloaded: $PRETRAINED_WEIGHTS"
else
    echo "Pretrained weights found: $PRETRAINED_WEIGHTS"
fi

# Step 2: Download sample training data (optional - uses ProteinMPNN sample data)
echo ""
echo "Step 2: Checking for training data..."
if [ ! -d "$DATA_DIR/pdbs" ] || [ -z "$(ls -A $DATA_DIR/pdbs 2>/dev/null)" ]; then
    echo "Downloading sample PDB data..."
    mkdir -p "$DATA_DIR/pdbs"
    
    # Download a few example PDBs from RCSB
    # Replace these with your own PDB files
    for pdb in "1qys" "5l33" "3htb"; do
        wget -q "https://files.rcsb.org/download/${pdb}.pdb" -O "$DATA_DIR/pdbs/${pdb}.pdb" || \
            echo "Could not download ${pdb}.pdb"
    done
    echo "Downloaded sample PDBs to $DATA_DIR/pdbs/"
else
    echo "Training data found in $DATA_DIR/pdbs/"
fi

# Count PDB files
NUM_PDBS=$(ls -1 "$DATA_DIR/pdbs/"*.pdb 2>/dev/null | wc -l)
echo "Found $NUM_PDBS PDB files"

if [ "$NUM_PDBS" -eq 0 ]; then
    echo "ERROR: No PDB files found. Please add PDB files to $DATA_DIR/pdbs/"
    echo "You can use your own protein structures or download from RCSB PDB"
    exit 1
fi

# Step 3: Prepare data
echo ""
echo "Step 3: Preparing data..."
python prepare_data.py \
    --pdb_dir "$DATA_DIR/pdbs" \
    --output_dir "$DATA_DIR/prepared" \
    --val_ratio 0.2 \
    --seed 42

# Step 4: Create fixed positions config (example)
echo ""
echo "Step 4: Creating example fixed positions config..."
cat > "$DATA_DIR/fixed_positions.json" << 'EOF'
{
    "_comment": "Example fixed positions configuration",
    "_note": "Modify this file for your specific use case"
}
EOF
echo "Created $DATA_DIR/fixed_positions.json (empty template)"

# Step 5: Fine-tune the model
echo ""
echo "Step 5: Fine-tuning ProteinMPNN..."
echo "This may take a while depending on your GPU..."

python finetune_proteinmpnn.py \
    --data_dir "$DATA_DIR/prepared/train" \
    --val_data_dir "$DATA_DIR/prepared/val" \
    --checkpoint "$PRETRAINED_WEIGHTS" \
    --output_dir "$OUTPUT_DIR/finetuned_model" \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-4 \
    --backbone_noise 0.1 \
    --log_interval 5 \
    --eval_interval 1 \
    --save_interval 5

# Step 6: Sample sequences
echo ""
echo "Step 6: Sampling sequences from fine-tuned model..."

# Use first PDB file for sampling demo
SAMPLE_PDB=$(ls "$DATA_DIR/pdbs/"*.pdb | head -1)

if [ -f "$OUTPUT_DIR/finetuned_model/checkpoints/checkpoint_best.pt" ]; then
    CHECKPOINT="$OUTPUT_DIR/finetuned_model/checkpoints/checkpoint_best.pt"
else
    CHECKPOINT="$OUTPUT_DIR/finetuned_model/checkpoints/checkpoint_latest.pt"
fi

python sample_sequences.py \
    --checkpoint "$CHECKPOINT" \
    --pdb_path "$SAMPLE_PDB" \
    --output_dir "$OUTPUT_DIR/designed_sequences" \
    --num_samples 10 \
    --temperature "0.1 0.2"

# Done
echo ""
echo "=============================================="
echo "Workflow complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Fine-tuned model: $OUTPUT_DIR/finetuned_model/"
echo "  - Designed sequences: $OUTPUT_DIR/designed_sequences/"
echo ""
echo "To monitor training with TensorBoard:"
echo "  tensorboard --logdir $OUTPUT_DIR/finetuned_model/tensorboard"
echo ""
echo "To design more sequences:"
echo "  python sample_sequences.py \\"
echo "    --checkpoint $CHECKPOINT \\"
echo "    --pdb_path YOUR_PDB.pdb \\"
echo "    --num_samples 100"
