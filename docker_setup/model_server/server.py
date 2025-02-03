from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import yaml
import os
from model.model import SmolLM2, SmolLM2Config
from typing import Optional

app = FastAPI(title="SmolLM2 API Server")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.8

class GenerationResponse(BaseModel):
    generated_text: str

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def init_model(checkpoint_path: str, config_path: str):
    config = load_config(config_path)
    
    if 'layer_norm_epsilon' in config['model']:
        config['model']['layer_norm_epsilon'] = float(config['model']['layer_norm_epsilon'])
    
    model_config = SmolLM2Config(**config['model'])
    model = SmolLM2(model_config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model = model.to(dtype=torch.bfloat16, device='cuda')
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return model, tokenizer

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    checkpoint_path = "/app/model/checkpoints/step_5051.pt"
    config_path = "/app/model/config.yaml"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this server")
    
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Your GPU does not support bfloat16")
    
    print("Loading model and tokenizer...")
    model, tokenizer = init_model(checkpoint_path, config_path)
    print("Model loaded successfully!")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        input_ids = tokenizer.encode(request.prompt, return_tensors='pt').to(device='cuda')
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return GenerationResponse(generated_text=generated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 