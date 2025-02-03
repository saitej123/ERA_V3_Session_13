import torch
import gradio as gr
from transformers import AutoTokenizer
from model.model import SmolLM2, SmolLM2Config
import yaml

# Load model configuration
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Initialize model and tokenizer
def init_model(checkpoint_path: str, config_path: str):
    config = load_config(config_path)
    model_config = SmolLM2Config(**config['model'])
    model = SmolLM2(model_config)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Use GPT-2 tokenizer since the model uses the same vocabulary
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return model, tokenizer

def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.8, model=None, tokenizer=None):
    if not prompt:
        return "Please provide a prompt!"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Initialize model and tokenizer
checkpoint_path = "ERA_V3_Session_13/checkpoints/step_5051.pt"
config_path = "config.yaml"  # Make sure this exists
model, tokenizer = init_model(checkpoint_path, config_path)

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

if __name__ == "__main__":
    demo.launch(share=True) 