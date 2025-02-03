#!/bin/bash

# Exit on error and print commands
set -ex

# Create directories if they don't exist
mkdir -p checkpoints
mkdir -p logs

# Function to check if a command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Check if huggingface-cli is logged in
echo "Checking Hugging Face login status..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "Please login to Hugging Face first using:"
    echo "huggingface-cli login"
    exit 1
fi

# Initial training for 5000 steps
echo "Starting initial training for 5000 steps..."
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/model_config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 5000 2>&1 | tee logs/initial_training.log
check_status "Initial training"

# Verify first checkpoint exists
if [ ! -f "checkpoints/step_5000.pt" ]; then
    echo "Error: Initial training checkpoint (step_5000.pt) not found!"
    # Try to find the latest checkpoint before 5000
    LATEST_CHECKPOINT=$(ls -t checkpoints/step_*.pt 2>/dev/null | head -n1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Found latest checkpoint: $LATEST_CHECKPOINT"
        FIRST_CHECKPOINT=$LATEST_CHECKPOINT
    else
        echo "No checkpoints found!"
        exit 1
    fi
else
    FIRST_CHECKPOINT="checkpoints/step_5000.pt"
fi

# Additional training for 50 steps from checkpoint
echo "Starting additional training for 50 steps from $FIRST_CHECKPOINT..."
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/model_config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 50 \
    --checkpoint_path "$FIRST_CHECKPOINT" 2>&1 | tee logs/additional_training.log
check_status "Additional training"

# Calculate expected step number from checkpoint name
STEP_NUM=$(echo "$FIRST_CHECKPOINT" | grep -o '[0-9]\+' | head -n1)
EXPECTED_FINAL_STEP=$((STEP_NUM + 50))
FINAL_CHECKPOINT="checkpoints/step_${EXPECTED_FINAL_STEP}.pt"

# Verify final checkpoint exists
if [ ! -f "$FINAL_CHECKPOINT" ]; then
    echo "Warning: Expected checkpoint $FINAL_CHECKPOINT not found!"
    # Try to find the latest checkpoint
    LATEST_CHECKPOINT=$(ls -t checkpoints/step_*.pt 2>/dev/null | head -n1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Using latest available checkpoint: $LATEST_CHECKPOINT"
        FINAL_CHECKPOINT=$LATEST_CHECKPOINT
    else
        echo "No checkpoints found!"
        exit 1
    fi
fi

# Upload model to Hugging Face Hub
echo "Uploading model to Hugging Face Hub..."
CUDA_VISIBLE_DEVICES=0 python upload_to_hub.py \
    --model_path "$FINAL_CHECKPOINT" \
    --config_path config/model_config.yaml \
    --repo_name "Saiteja/smollm2-135m" \
    --total_steps "$EXPECTED_FINAL_STEP"
check_status "Model upload"

echo "Training and upload complete!"
echo "Final checkpoint: $FINAL_CHECKPOINT"
echo "Model uploaded to: https://huggingface.co/Saiteja/smollm2-135m" 