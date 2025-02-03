# SmolLM2 Training

This repository contains the code and configuration for training the SmolLM2 language model.

## Setup

1. Install dependencies:
```bash
pip install torch transformers accelerate wandb pyyaml
```

2. Set up WandB:
```bash
wandb login
```

3. Prepare your training data:
- Place your training text in `input.txt`
- The text should be raw text format
- The model will automatically handle tokenization

4. Configure the model:
- Edit `config.yaml` to adjust model and training parameters
- The configuration is split into three main sections:
  - `model`: Architecture and model parameters
  - `training`: Training process parameters
  - `wandb`: Logging and visualization settings

## Training

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

## Configuration

### Model Parameters
- `hidden_dim`: Size of hidden layers (768)
- `n_layers`: Number of transformer layers (12)
- `n_heads`: Number of attention heads (12)
- `intermediate_size`: Size of feedforward layers (3072)
- `max_position_embeddings`: Maximum sequence length (2048)
- `vocab_size`: Size of vocabulary (32000)

### Training Parameters
- `learning_rate`: Learning rate (1e-4)
- `weight_decay`: Weight decay for AdamW (0.01)
- `batch_size`: Batch size per GPU (12)
- `gradient_accumulation_steps`: Steps before optimizer update (4)
- `eval_steps`: Steps between evaluations (100)
- `save_steps`: Steps between checkpoints (500)

### WandB Configuration
- Logging Options:
  - Model checkpoints
  - Gradients and parameters
  - Batch metrics
  - Generated text samples
  - Model parameters
- Artifacts:
  - Code snapshots
  - Checkpoints
- Visualizations:
  - Learning rate curves
  - Gradient flow
  - 3D loss landscape

## Monitoring Training

1. Real-time Monitoring:
- Open your WandB dashboard to view:
  - Training loss
  - Learning rate
  - Generated samples
  - Model gradients
  - System metrics

2. Visualizations:
- Learning curves
- Gradient flow charts
- Loss landscape analysis
- Parameter distributions

## Checkpoints

Checkpoints are saved in the `checkpoints` directory with the format:
```
checkpoints/step_XXXX.pt
```

Each checkpoint contains:
- Model state
- Optimizer state
- Scheduler state
- Training step
- Last loss value

All checkpoints are automatically logged to WandB as artifacts.

## Logs

Training logs are tracked in WandB and include:
- Loss values
- Learning rates
- Generated samples
- Error messages (if any)
- System metrics (GPU usage, memory, etc.)
- Model gradients and parameters

## Notes

- The training script uses gradient checkpointing to reduce memory usage
- Mixed precision training is temporarily disabled for debugging
- The model uses layer normalization for stability
- Attention masks are properly handled for causal language modeling
- All training metrics and artifacts are logged to WandB for analysis 