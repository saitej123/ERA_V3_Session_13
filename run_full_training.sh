#!/bin/bash

# Set environment variables for better performance
export TORCH_USE_CUDA_DSA=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Run initial training for 5000 steps
echo "Starting initial training phase (5000 steps)..."
python train.py \
    --config config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 5000 \
    --eval_steps 100 \
    --save_steps 500 \
    --prediction_steps 200

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Initial training completed successfully. Starting continued training..."
    
    # Continue training for 50 more steps
    python continue_training.py \
        --config config.yaml \
        --input_file input.txt \
        --save_dir checkpoints
else
    echo "Initial training failed. Please check the logs."
    exit 1
fi 