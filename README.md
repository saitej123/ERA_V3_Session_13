# SmolLM2-135: A Compact and Efficient Language Model

SmolLM2-135 is a compact yet powerful language model with approximately 123.6M parameters, designed for efficient training and inference while maintaining strong performance.

## Model Architecture

### Core Components
- **Hidden Dimension**: 768
- **Number of Layers**: 12
- **Attention Heads**: 12 (64 dimensions per head)
- **Intermediate Size**: 3072 (4x hidden dimension)
- **Max Sequence Length**: 2048
- **Vocabulary Size**: 50,258 (after tokenizer)

### Parameter Breakdown
1. **Token Embeddings**: 38.6M parameters
   - Embedding matrix: 50,258 × 768

2. **Transformer Layers** (×12): 84.97M parameters per layer
   Each layer contains:
   - Multi-Head Self-Attention
     - Q/K/V projections: 1.77M
     - Output projection: 0.59M
   - MLP
     - First layer: 2.36M
     - Second layer: 2.36M
   - Layer Normalization: 3.07K

3. **Final Layer Norm**: 1.5K parameters

Total Parameters: ~123.6M

### Advanced Features
1. **Flash Attention**
   - Efficient attention computation
   - Reduced memory footprint
   - Faster training and inference

2. **Rotary Embeddings**
   - Better position encoding
   - Improved relative position modeling
   - Enhanced sequence understanding

3. **Mixed Precision Training**
   - BF16 precision for faster training
   - Reduced memory usage
   - Maintained numerical stability

4. **Gradient Checkpointing**
   - Memory-efficient backpropagation
   - Enables training with longer sequences
   - Trades computation for memory

## Training Process

### Environment Setup
```bash
# Performance Optimizations
export TORCH_USE_CUDA_DSA=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# WandB Configuration
export WANDB_PROJECT="smollm2-training"
export WANDB_WATCH="gradients"
export WANDB_LOG_MODEL="true"
export WANDB_SILENT="false"
export WANDB_CONSOLE="wrap"
export WANDB_DISABLE_CODE="false"
```

### Training Configuration
- **Phase 1**: Initial Training
  - Steps: 5000
  - Batch Size: 48 (optimized for balanced throughput and memory usage)
  - Gradient Accumulation Steps: 4
  - Evaluation & Logging: Every 500 steps
  - Checkpoint & Generation: Every 500 steps
  - Mixed Precision: BF16

- **Phase 2**: Continuous Training
  - Steps: 50 (5000-5050)
  - Maintains all hyperparameters
  - Loads checkpoint from step 5000
  - Continuous WandB logging

### Quick Start
```bash
# Make the training script executable
chmod +x train.sh

# Start training
./train.sh
```

### Training Strategy
1. **Initial Training Phase (0-5000 steps)**
   ```bash
   python train.py \
       --config config.yaml \
       --input_file input.txt \
       --save_dir checkpoints \
       --train_steps 5000
   ```
   - Mixed precision (BF16)
   - Flash Attention enabled
   - Regular checkpointing
   - WandB logging with samples

2. **Extended Training Phase (5000-5050 steps)**
   ```bash
   python continue_training.py \
       --config config.yaml \
       --input_file input.txt \
       --save_dir checkpoints
   ```
   - Seamless continuation
   - State preservation
   - Continuous logging

### Optimization Parameters
- AdamW optimizer
- Learning rate: 5e-4
- Weight decay: 0.1
- Gradient clipping: 1.0
- Batch size: 48
- Gradient accumulation steps: 4

### Generation Settings
- Temperature: 1.0
- Top-k: 50
- Top-p: 0.95
- Max length: 100
- Number of samples: 3
- Prompt templates:
  ```python
  [
    "Once upon a time",
    "In a world where",
    "The most interesting thing about"
  ]
  ```

## Monitoring and Logging

### WandB Integration
1. **Metrics Tracking**:
   - Loss
   - Learning rate
   - Perplexity
   - Generated tokens

2. **Sample Logging**:
   - Generation frequency: Every 500 steps
   - Samples per generation: 3
   - Tracked metrics:
     - Generation time
     - Perplexity
     - Sample length

3. **Visualizations**:
   - Loss curve
   - Learning rate curve
   - Gradient flow
   - Sample length distribution

4. **Artifacts**:
   - Model checkpoints
   - Generated samples table
   - Training logs
   - Code snapshots

### Local Logging
- Detailed logs in `logs/` directory
  - Phase 1: `logs/phase1.log`
  - Phase 2: `logs/phase2.log`
- Checkpoints in `checkpoints/` directory
- Automatic log rotation and management

## Error Handling
- Phase completion verification
- Checkpoint validation
- Training state preservation
- Detailed error logging
- Graceful interruption handling

## Model Weights
- [Lightning AI](https://lightning.ai/saitej/studios/era-session-13~01jhx4rv2zfkfn1twcb5ewarr9?view=public&section=featured&tab=files&layout=column&path=cloudspaces%2F01jk5xqk86shag2kx1jew0mbz0)
- [Weights and bias](https://wandb.ai/macharlasaiteja/smollm2-training?nw=nwusermacharlasaiteja)

## Citations
```bibtex
@misc{smollm2-135,
  author = {Saiteja},
  title = {SmolLM2-135: A Compact and Efficient Language Model},
  year = {20245},
  publisher = {GitHub},
  url = {https://github.com/saitej123}
}
``` 