import torch
import gradio as gr
from transformers import AutoTokenizer
from model.model import SmolLM2, SmolLM2Config
import yaml
import os

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is required for this demo"

# Load model configuration
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Initialize model and tokenizer
def init_model(checkpoint_path: str, config_path: str):
    config = load_config(config_path)
    
    # Ensure layer_norm_epsilon is float
    if 'layer_norm_epsilon' in config['model']:
        config['model']['layer_norm_epsilon'] = float(config['model']['layer_norm_epsilon'])
    
    model_config = SmolLM2Config(**config['model'])
    model = SmolLM2(model_config)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    # Convert model to bf16 and move to CUDA
    model = model.to(dtype=torch.bfloat16, device='cuda')
    
    # Use GPT-2 tokenizer since the model uses the same vocabulary
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return model, tokenizer

def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.8, model=None, tokenizer=None):
    if not prompt:
        return "Please provide a prompt!"
    
    # Keep input_ids as Long type and move to CUDA
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda')
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else None,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    # Initialize model and tokenizer
    checkpoint_path = "./checkpoints/step_5051.pt"
    config_path = "config.yaml"
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Check if bf16 is supported
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Your GPU does not support bfloat16. Please use a newer GPU.")
    
    print("Loading model and tokenizer...")
    model, tokenizer = init_model(checkpoint_path, config_path)
    print("Model loaded successfully!")

    # Create Gradio interface
    def gradio_interface(prompt, max_length, temperature):
        return generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            model=model,
            tokenizer=tokenizer
        )

    # Create and launch the interface
    demo = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
            gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Max Length"),
            gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        ],
        outputs=gr.Textbox(label="Generated Text"),
        title="SmolLM2 Text Generation",
        description="Enter a prompt and the model will generate text based on it. Adjust max length and temperature to control the generation.",
    )
    
    print("Starting Gradio interface...")
    demo.launch(share=True)

if __name__ == "__main__":
    main() 