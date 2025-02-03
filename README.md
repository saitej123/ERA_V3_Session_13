# SmolLM2 Training

This repository contains the code and configuration for training the SmolLM2 language model, a transformer-based model with flash attention and mixed precision training.

## Features

- Flash Attention for efficient attention computation
- Mixed precision training (automatic bf16/fp16 selection)
- Gradient checkpointing for memory efficiency
- Rotary positional embeddings
- WandB integration for experiment tracking
- Robust error handling and validation

## Setup

1. Install dependencies:
```bash
pip install torch transformers accelerate wandb flash-attn rotary-embedding-torch pyyaml
```

2. Set up WandB:
```bash
wandb login
```

3. Prepare your training data:
- Place your training text in `input.txt`
- The text will be automatically tokenized and processed in chunks
- Supports long sequences with proper truncation and stride

4. Configure the model:
Edit `config.yaml` to adjust parameters in three main sections:
- `model`: Architecture and features
- `training`: Training process and optimization
- `wandb`: Logging and visualization

## Model Architecture

### Core Parameters
- Hidden dimension: 768
- Number of layers: 12
- Number of heads: 12
- Intermediate size: 3072
- Maximum sequence length: 2048
- Vocabulary size: 32000 (auto-adjusted based on tokenizer)

### Features
- Flash Attention support
- Rotary positional embeddings
- Gradient checkpointing
- Layer normalization with stability improvements

## Training Configuration

### Optimization
- Learning rate: 1e-4
- Weight decay: 0.01
- Gradient clipping: 1.0
- Batch size: 12
- Gradient accumulation steps: 4

### Mixed Precision
- Automatic dtype selection (bf16 if supported, otherwise fp16)
- Improved numerical stability in layer normalization
- Proper dtype handling throughout the model

### Training Process
- Evaluation every 100 steps
- Checkpoints every 500 steps
- Text generation samples every 200 steps
- Cosine learning rate schedule
- Warm-up steps: 100

## WandB Integration

### Logging Features
- Model checkpoints
- Training metrics
- Generated text samples
- Model parameters
- Gradient flow
- Learning rate curves

### Artifacts
- Code snapshots
- Checkpoints
- Configuration files

### Visualizations
- Learning rate curves
- Gradient flow charts
- Loss landscapes
- Training metrics

## Usage

1. Start training:
```bash
./run_training.sh
```

2. Resume from checkpoint:
```bash
python train.py \
    --config config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 5000 \
    --checkpoint_path checkpoints/step_XXXX.pt
```

3. Disable WandB logging:
```bash
python train.py \
    --config config.yaml \
    --input_file input.txt \
    --save_dir checkpoints \
    --train_steps 5000 \
    --disable_wandb
```

## Checkpoints

Checkpoints are saved with the following information:
- Model state
- Optimizer state
- Scheduler state
- Current step
- Current loss
- Format: `checkpoints/step_XXXX.pt`

## Safety Features

1. Input Validation:
- Token index validation
- Proper handling of unknown tokens
- Vocabulary size checks
- Tensor shape validation

2. Training Stability:
- Gradient clipping
- Mixed precision training
- Improved layer normalization
- Proper device placement

3. Error Handling:
- Checkpoint saving protection
- Dataset validation
- Proper cleanup on interruption
- Detailed error logging

## Monitoring

1. Console Logging:
- Training progress
- Loss values
- Learning rates
- Error messages
- Dataset statistics

2. WandB Dashboard:
- Real-time metrics
- Generated samples
- System metrics
- Model gradients
- Training curves

## Notes

- Uses gradient checkpointing to reduce memory usage
- Automatically selects optimal mixed precision format
- Handles long sequences with proper chunking
- Includes robust error handling and validation
- Supports distributed training through Accelerate
- All metrics and artifacts are logged to WandB 