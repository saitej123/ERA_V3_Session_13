#!/bin/bash

# Create directories
mkdir -p checkpoints
mkdir -p logs

# Initial training for 5000 steps
echo "Starting initial training for 5000 steps..."
python train.py \
    --config config/model_config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 5000 2>&1 | tee logs/initial_training.log

# Additional training for 50 steps from checkpoint
echo "Starting additional training for 50 steps..."
python train.py \
    --config config/model_config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 50 \
    --checkpoint_path checkpoints/step_5000.pt 2>&1 | tee logs/additional_training.log

# Upload model to Hugging Face Hub
echo "Uploading model to Hugging Face Hub..."
python upload_to_hub.py \
    --model_path checkpoints/step_5050.pt \
    --config_path config/model_config.yaml \
    --repo_name $HF_USERNAME/smollm2-135m \
    --token $HF_TOKEN \
    --total_steps 5050

echo "Training and upload complete!" 