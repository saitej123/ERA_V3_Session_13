#!/bin/bash

# Exit on error and print commands
set -ex

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Set environment variables for better performance and debugging
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1  # Enable device-side assertions
export TORCH_SHOW_CPP_STACKTRACES=1  # Show C++ stack traces
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Reduce concurrent CUDA operations

# Configure WandB (modify these as needed)
export WANDB_PROJECT="smollm2-training"
export WANDB_ENTITY="macharlasaiteja"  # Your WandB username
export WANDB_NAME="smollm2-training-run"
export WANDB_TAGS="[smollm2, training]"

# Ensure WandB is logged in
if ! wandb status >/dev/null 2>&1; then
    echo "Please log in to WandB first using: wandb login"
    exit 1
fi

# Run training with WandB logging enabled
python train.py \
    --config config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 5000

echo "Training complete! Check WandB dashboard for logs and visualizations." 