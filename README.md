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

### Configuration
- Gradient Accumulation Steps: 4
- Training Steps: 5000 + 50
- Warmup Steps: 100
- Evaluation Frequency: Every 100 steps
- Checkpoint Saving: Every 500 steps
- Text Generation: Every 200 steps

### Training Strategy
1. Initial Training Phase (5000 steps)
   - Mixed precision training (BF16)
   - Flash Attention for efficient computation
   - Regular evaluation and checkpoint saving

2. Extended Training Phase (50 steps)
   - Loaded from step 5000 checkpoint
   - Continued training with same configuration
   - Demonstrated checkpoint restoration capability

### Optimization
- AdamW optimizer
- Learning rate: 5e-4
- Weight decay: 0.1
- Gradient clipping: 1.0

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
   - Optimized attention computation
   - Efficient data loading and processing
   - Parallel computation with multiple GPUs

3. **Generation Quality**
   - Top-k and nucleus sampling
   - Temperature control for diversity
   - EOS token handling for proper completion

## Monitoring and Logging
- WandB integration for:
  - Loss tracking
  - Learning rate scheduling
  - Generated text samples
  - Resource utilization
  - Training progress

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