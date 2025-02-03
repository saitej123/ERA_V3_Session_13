import argparse
import os
import yaml
import math
import logging
from pathlib import Path
from typing import Optional
from dataclasses import fields

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from transformers import AutoTokenizer
import wandb

from model.model import SmolLM2, SmolLM2Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, block_size: int = 1024):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        logger.info(f"Tokenizing text from {file_path}")
        self.examples = tokenizer(
            text,
            truncation=True,
            max_length=block_size,
            return_tensors="pt",
            return_overflowing_tokens=True,
            padding="max_length",
        )["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_config(config_path: str) -> dict:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure training parameters are of correct type
    training_config = config.get('training', {})
    training_config['learning_rate'] = float(training_config.get('learning_rate', 1e-4))
    training_config['weight_decay'] = float(training_config.get('weight_decay', 0.01))
    training_config['max_grad_norm'] = float(training_config.get('max_grad_norm', 1.0))
    training_config['batch_size'] = int(training_config.get('batch_size', 32))
    training_config['gradient_accumulation_steps'] = int(training_config.get('gradient_accumulation_steps', 4))
    training_config['eval_steps'] = int(training_config.get('eval_steps', 500))
    training_config['save_steps'] = int(training_config.get('save_steps', 500))
    training_config['prediction_steps'] = int(training_config.get('prediction_steps', 500))
    training_config['max_steps'] = int(training_config.get('max_steps', 5000))
    training_config['warmup_steps'] = int(training_config.get('warmup_steps', 100))
    
    # Ensure model parameters are of correct type
    model_config = config.get('model', {})
    model_config['hidden_dim'] = int(model_config.get('hidden_dim', 768))
    model_config['n_layers'] = int(model_config.get('n_layers', 12))
    model_config['n_heads'] = int(model_config.get('n_heads', 12))
    model_config['intermediate_size'] = int(model_config.get('intermediate_size', 3072))
    model_config['max_position_embeddings'] = int(model_config.get('max_position_embeddings', 2048))
    model_config['vocab_size'] = int(model_config.get('vocab_size', 32000))
    model_config['dropout'] = float(model_config.get('dropout', 0.1))
    model_config['attention_dropout'] = float(model_config.get('attention_dropout', 0.1))
    model_config['layer_norm_epsilon'] = float(model_config.get('layer_norm_epsilon', 1e-5))
    model_config['initializer_range'] = float(model_config.get('initializer_range', 0.02))
    
    config['training'] = training_config
    config['model'] = model_config
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    loss: float,
    save_dir: str,
):
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': step,
            'loss': loss,
        }
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f'step_{step}.pt')
        
        # Save to a temporary file first
        temp_path = checkpoint_path + '.tmp'
        torch.save(checkpoint, temp_path)
        
        # Rename the temporary file to the final name
        os.replace(temp_path, checkpoint_path)
        
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint at step {step}: {str(e)}")
        return False


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step'], checkpoint['loss']


def generate_sample(model: nn.Module, tokenizer, prompt: str = "Once upon a time", max_length: int = 100):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def train(args):
    # Initialize wandb
    wandb.init(project="smollm2-training", config=args)

    # Load configuration
    config = load_config(args.config)
    
    # Filter out any config keys that aren't in SmolLM2Config
    valid_config_keys = {f.name for f in fields(SmolLM2Config)}
    model_config_dict = {k: v for k, v in config['model'].items() if k in valid_config_keys}
    model_config = SmolLM2Config(**model_config_dict)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='bf16' if config['training']['bf16'] else 'fp16',
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataset and dataloader
    dataset = TextDataset(args.input_file, tokenizer)
    dataloader = create_dataloader(dataset, config['training']['batch_size'])

    # Initialize model, optimizer, and scheduler
    model = SmolLM2(model_config)
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.train_steps,
        eta_min=1e-5,
    )

    # Load checkpoint if specified
    start_step = 0
    if args.checkpoint_path:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
        start_step, _ = load_checkpoint(args.checkpoint_path, model, optimizer, scheduler)
        logger.info(f"Loaded checkpoint from step {start_step}")

    # Prepare for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Training loop
    model.train()
    data_iter = iter(dataloader)
    total_steps = start_step + args.train_steps
    step = start_step
    last_save_successful = True

    try:
        while step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            with accelerator.accumulate(model):
                outputs, loss = model(batch, labels=batch)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % config['training']['eval_steps'] == 0:
                avg_loss = accelerator.gather(loss).mean().item()
                logger.info(f"Step {step}: loss = {avg_loss:.4f}")
                wandb.log({
                    "loss": avg_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": step,
                })

            if step % config['training']['prediction_steps'] == 0:
                sample = generate_sample(model, tokenizer)
                logger.info(f"\nGenerated sample at step {step}:\n{sample}\n")
                wandb.log({"generated_text": sample, "step": step})

            if step % config['training']['save_steps'] == 0 or step == total_steps - 1:
                last_save_successful = save_checkpoint(
                    accelerator.unwrap_model(model),
                    optimizer,
                    scheduler,
                    step + 1,  # Save with the next step number
                    loss.item(),
                    args.save_dir,
                )
                if not last_save_successful:
                    logger.warning("Failed to save checkpoint, but continuing training...")

            step += 1

    except Exception as e:
        logger.error(f"Training interrupted at step {step}: {str(e)}")
        # Try to save a final checkpoint if the last save wasn't successful
        if not last_save_successful:
            save_checkpoint(
                accelerator.unwrap_model(model),
                optimizer,
                scheduler,
                step,
                loss.item(),
                args.save_dir,
            )
        raise

    # Save final checkpoint if we haven't just saved one
    if step % config['training']['save_steps'] != 0:
        save_checkpoint(
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
            step,
            loss.item(),
            args.save_dir,
        )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input text file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--train_steps", type=int, required=True, help="Number of training steps")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    try:
        train(args)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise 