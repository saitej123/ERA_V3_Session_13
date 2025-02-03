import os
import argparse
from pathlib import Path
import shutil
import yaml
import json
from huggingface_hub import HfApi, create_repo, whoami

def prepare_model_card(config_path: str, training_args: dict) -> str:
    """Create a model card for the Hugging Face Hub."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_card = f"""---
language: en
tags:
- pytorch
- causal-lm
- language-model
- smollm2
license: mit
---

# SmolLM2-135M Language Model

This is an implementation of SmolLM2-135M, a lightweight language model with 135M parameters.

## Model Description

- **Model Type:** Causal Language Model
- **Language:** English
- **Total Parameters:** 135M
- **Architecture:**
  - {config['model']['n_layers']} transformer layers
  - {config['model']['hidden_dim']} hidden dimension
  - {config['model']['n_heads']} attention heads
  - {config['model']['intermediate_size']} intermediate size (FFN)
  - Layer normalization before attention and FFN (Pre-LN architecture)
  - Rotary positional embeddings
  - Vocabulary size: {config['model']['vocab_size']}

## Training Details

- **Training Steps:** {training_args['total_steps']}
- **Batch Size:** {config['training']['batch_size']}
- **Learning Rate:** {config['training']['learning_rate']}
- **Weight Decay:** {config['training']['weight_decay']}
- **Mixed Precision:** {'bf16' if config['training']['bf16'] else 'fp16'}
- **Gradient Checkpointing:** {config['model']['gradient_checkpointing']}
- **Flash Attention:** {config['model']['use_flash_attention']}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Saiteja/smollm2-135m")
tokenizer = AutoTokenizer.from_pretrained("Saiteja/smollm2-135m")

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

This model is trained for research and experimental purposes. It may exhibit biases and generate inappropriate content.

## License

This model is released under the MIT License.
"""
    return model_card

def upload_to_hub(
    model_path: str,
    config_path: str,
    repo_name: str,
    training_args: dict,
):
    """Upload the model, tokenizer, and configs to the Hugging Face Hub."""
    # Verify HF authentication
    try:
        user = whoami()
        print(f"Logged in to Hugging Face as: {user['name']}")
    except Exception as e:
        raise RuntimeError("Please login using 'huggingface-cli login' first") from e

    # Create the repository
    api = HfApi()
    repo_url = create_repo(
        repo_name,
        private=False,
        exist_ok=True,
    )

    # Create a temporary directory for organizing files
    tmp_dir = Path("tmp_upload")
    tmp_dir.mkdir(exist_ok=True)

    try:
        # Copy model files
        shutil.copy2(model_path, tmp_dir / "pytorch_model.bin")
        shutil.copy2(config_path, tmp_dir / "config.yaml")

        # Create model card
        model_card = prepare_model_card(config_path, training_args)
        with open(tmp_dir / "README.md", "w") as f:
            f.write(model_card)

        # Create config.json for Hugging Face compatibility
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        hf_config = {
            "architectures": ["SmolLM2ForCausalLM"],
            "model_type": "smollm2",
            "torch_dtype": "bfloat16" if config['training']['bf16'] else "float16",
            **config['model']
        }
        
        with open(tmp_dir / "config.json", "w") as f:
            json.dump(hf_config, f, indent=2)

        # Upload everything to the Hub
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_name,
        )

        print(f"Successfully uploaded model to: https://huggingface.co/{repo_name}")

    finally:
        # Clean up
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--repo_name", type=str, required=True, help="Name of the Hugging Face repository")
    parser.add_argument("--total_steps", type=int, required=True, help="Total training steps")
    args = parser.parse_args()

    training_args = {
        "total_steps": args.total_steps,
    }

    upload_to_hub(
        model_path=args.model_path,
        config_path=args.config_path,
        repo_name=args.repo_name,
        training_args=training_args,
    ) 