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
```

### Training Configuration
- **Phase 1**: Initial Training
  - Steps: 5000
  - Evaluation Frequency: Every 100 steps
  - Checkpoint Saving: Every 500 steps
  - Text Generation: Every 200 steps
  - Mixed Precision: BF16
  - Gradient Accumulation Steps: 4

- **Phase 2**: Continuous Training
  - Steps: 50 (5000-5050)
  - Loads checkpoint from step 5000
  - Maintains all hyperparameters
  - Continues WandB logging
  - Preserves optimizer state

### Quick Start
```bash
# Make the training script executable
chmod +x train.sh

# Start training
./train.sh
```

### Training Strategy
1. **Initial Training Phase (0-5000 steps)**
   - Mixed precision training (BF16)
   - Flash Attention for efficient computation
   - Regular evaluation and checkpointing
   - Progress tracking in WandB
   - Automatic checkpoint at step 5000

2. **Extended Training Phase (5000-5050 steps)**
   - Seamless checkpoint loading
   - Continuous training without parameter reset
   - Maintained optimization state
   - Uninterrupted WandB logging
   - Final checkpoint at step 5050

### Optimization Parameters
- AdamW optimizer
- Learning rate: 5e-4
- Weight decay: 0.1
- Gradient clipping: 1.0
- Gradient accumulation steps: 4

### Generation Settings
- Temperature: 1.0
- Top-k: 50
- Top-p: 0.95
- Max length: 100

## Performance Features

1. **Memory Efficiency**
   - Flash Attention for O(n) memory complexity
   - Gradient checkpointing for reduced memory usage
   - Mixed precision training for memory optimization

2. **Training Speed**
   - Optimized CUDA operations
   - Efficient data loading and processing
   - Parallel computation with multiple GPUs
   - Environment-level optimizations

3. **Generation Quality**
   - Top-k and nucleus sampling
   - Temperature control for diversity
   - EOS token handling for proper completion

## Monitoring and Logging
- **WandB Integration**:
  - Loss tracking
  - Learning rate scheduling
  - Generated text samples
  - Resource utilization
  - Training progress
  - Gradient flow visualization
  - Model checkpoints
  - Continuous run tracking

## Error Handling
- Phase completion verification
- Checkpoint validation
- Training state preservation
- Detailed error logging
- Graceful interruption handling

## Model Weights
- [🤗 Hugging Face Space](link-to-your-space)
- [GitHub Repository](link-to-your-repo)

## Citations
```bibtex
@misc{smollm2-135,
  author = {Your Name},
  title = {SmolLM2-135: A Compact and Efficient Language Model},
  year = {2024},
  publisher = {GitHub},
  url = {link-to-your-repo}
}
``` 