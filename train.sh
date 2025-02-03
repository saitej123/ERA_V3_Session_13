#!/bin/bash

# Performance optimization environment variables
export TORCH_USE_CUDA_DSA=1  # Enable device-side assertions
export TORCH_SHOW_CPP_STACKTRACES=1  # Better error reporting
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Optimize CUDA operations
export TORCH_DISTRIBUTED_DEBUG=INFO  # Distributed training debug info
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA execution for better debugging

# Wandb setup
export WANDB_PROJECT="smollm2-training"
export WANDB_WATCH="gradients"
export WANDB_LOG_MODEL="true"
export WANDB_SILENT="false"  # Enable verbose logging
export WANDB_CONSOLE="wrap"  # Ensure console logging works with WandB
export WANDB_DISABLE_CODE="false"  # Track code changes

echo "Starting SmolLM2-135 training pipeline..."

# Create directories
mkdir -p checkpoints
mkdir -p logs

# Phase 1: Initial training (5000 steps)
echo "Phase 1: Starting initial training for 5000 steps..."
python train.py \
    --config config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 5000 2>&1 | tee logs/phase1.log

# Check if Phase 1 completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Phase 1 training failed. Check logs/phase1.log for details."
    exit 1
fi

echo "Phase 1 completed successfully!"

# Phase 2: Extended training (50 steps)
echo "Phase 2: Starting extended training for 50 steps..."
python continue_training.py \
    --config config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 50 \
    --start_step 5000 2>&1 | tee logs/phase2.log

# Check if Phase 2 completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Phase 2 training failed. Check logs/phase2.log for details."
    exit 1
fi

echo "Phase 2 completed successfully!"
echo "Full training pipeline completed! Total steps: 5050"

# Sync final WandB logs
wandb sync --sync-all

# Optional: Upload to Hugging Face Hub (uncomment and configure if needed)
# python upload_to_hub.py \
#     --model_path checkpoints/step_5050.pt \
#     --hub_model_id your-username/smollm2-135 