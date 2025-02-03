# SmolLM2-135 Implementation and Training

This repository contains an implementation and training pipeline for SmolLM2-135, a lightweight language model. The model is trained for 5000 steps with intermediate predictions every 500 steps, followed by an additional 50 steps of training from a checkpoint.

## Model Architecture

SmolLM2-135 is a transformer-based language model with the following specifications:

- **Total Parameters**: 135M parameters
- **Architecture**:
  - 12 transformer layers
  - 768 hidden dimension
  - 12 attention heads
  - 3072 intermediate size (FFN)
  - Layer normalization before attention and FFN (Pre-LN architecture)
  - Rotary positional embeddings
  - Vocabulary size: 32000 (using BPE tokenization)

### Parameter Calculation

Total parameters breakdown:
1. Token Embeddings: 768 * 32000 = 24.576M
2. Layer parameters (per layer):
   - Self-attention:
     - Q,K,V matrices: 3 * (768 * 768) = 1.769M
     - Output projection: 768 * 768 = 0.589M
   - FFN:
     - First projection: 768 * 3072 = 2.359M
     - Second projection: 3072 * 768 = 2.359M
   - Layer norms: 2 * 768 * 2 = 0.003M
   Total per layer: ~7.079M

Total parameters: 24.576M + (7.079M * 12) = ~135M parameters

## Project Structure

```
.
├── README.md
├── config
│   └── model_config.yaml
├── model
│   ├── model.py
│   └── utils.py
├── train.py
├── requirements.txt
└── input.txt
```

## Training Process

The training process consists of two phases:

1. **Initial Training (5000 steps)**:
   - Training with gradient checkpointing for memory efficiency
   - Flash Attention 2 for faster attention computation
   - Predictions generated every 500 steps
   - Checkpoint saved at step 5000

2. **Additional Training (50 steps)**:
   - Load checkpoint from step 5000
   - Continue training for 50 more steps
   - Final model saved and uploaded to Hugging Face Spaces

## Training Optimizations

The following optimizations are implemented for efficient training:

1. Gradient Checkpointing
2. Flash Attention 2
3. Mixed Precision Training (bfloat16)
4. Efficient Memory Management
5. Optimized DataLoader with prefetching

## Results

Training logs and intermediate predictions can be found in the model's Hugging Face Space:
[Link to be added after training]

## Model Links

- GitHub Repository: [Current Repository]
- Hugging Face Space: [To be added after training]

## Requirements

See `requirements.txt` for all dependencies. Key requirements:
- PyTorch >= 2.0
- Transformers
- Flash Attention 2
- Accelerate
- Wandb (for logging)

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initial training:
```bash
python train.py --config config/model_config.yaml --train_steps 5000
```

3. Continue training from checkpoint:
```bash
python train.py --config config/model_config.yaml --train_steps 50 --checkpoint_path checkpoints/step_5000.pt
``` 