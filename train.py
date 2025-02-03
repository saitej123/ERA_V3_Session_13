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

def parse_args():
    parser = argparse.ArgumentParser(description='Train SmolLM2')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--train_steps', type=int, default=None, help='Number of training steps')
    parser.add_argument('--start_step', type=int, default=0, help='Starting step for continued training')
    return parser.parse_args()

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Add this for better error messages
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, block_size: int = 1024):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        logger.info(f"Tokenizing text from {file_path}")
        
        # Process text in chunks to avoid memory issues
        chunk_size = 100000  # Process 100K characters at a time
        stride = block_size // 2  # Use stride for overlapping chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        all_input_ids = []
        vocab_size = tokenizer.vocab_size
        unk_token_id = tokenizer.unk_token_id
        
        # Add safety checks for special tokens
        if unk_token_id is None:
            unk_token_id = 0  # Fallback to 0 if no unk token
            logger.warning("No unk_token_id found, using 0 as fallback")
        
        for chunk in chunks:
            try:
                # Tokenize with explicit settings
                tokens = tokenizer(
                    chunk,
                    truncation=True,
                    max_length=block_size,
                    add_special_tokens=True,
                    padding=False,
                    return_tensors=None,
                )["input_ids"]
                
                # Ensure tokens is a flat list
                if isinstance(tokens[0], list):
                    tokens = [t for sublist in tokens for t in sublist]
                else:
                    tokens = list(tokens)
                
                # Validate and clip token indices
                for i, token_id in enumerate(tokens):
                    if not isinstance(token_id, int):
                        logger.warning(f"Non-integer token ID found: {token_id}, using unk_token")
                        tokens[i] = unk_token_id
                    elif token_id >= vocab_size:
                        logger.warning(f"Token ID {token_id} >= vocab_size {vocab_size}, using unk_token")
                        tokens[i] = unk_token_id
                    elif token_id < 0:
                        logger.warning(f"Negative token ID found: {token_id}, using unk_token")
                        tokens[i] = unk_token_id
                
                # Create overlapping chunks with validation
                for i in range(0, len(tokens) - block_size + 1, stride):
                    chunk = tokens[i:i + block_size]
                    if len(chunk) == block_size:
                        # Final validation before adding
                        if all(0 <= t < vocab_size for t in chunk):
                            all_input_ids.append(torch.tensor(chunk, dtype=torch.long))
                        else:
                            logger.warning(f"Invalid token IDs in chunk at position {i}, skipping")
                
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        if not all_input_ids:
            raise ValueError(f"No valid chunks created from {file_path}. Text might be too short or all chunks were invalid.")
        
        self.examples = torch.stack(all_input_ids)
        
        # Log dataset statistics
        max_token = max(max(x) for x in all_input_ids)
        min_token = min(min(x) for x in all_input_ids)
        logger.info(f"Created dataset with {len(self.examples)} examples of length {block_size}")
        logger.info(f"Token ID range: {min_token} to {max_token} (vocab size: {vocab_size})")
        logger.info(f"Memory usage: {self.examples.element_size() * self.examples.nelement() / 1024 / 1024:.2f} MB")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,  # Reduced number of workers
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
    # Initialize wandb (optional)
    if os.environ.get("WANDB_DISABLED", "").lower() != "true":
        wandb.init(project="smollm2-training", config=args)
    else:
        logger.info("WandB logging is disabled")

    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set default device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize accelerator with mixed precision settings
        mixed_precision = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp16'
        logger.info(f"Using mixed precision: {mixed_precision}")
        
        accelerator = Accelerator(
            mixed_precision=mixed_precision,  # Enable mixed precision training
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            log_with="wandb" if os.environ.get("WANDB_DISABLED", "").lower() != "true" else None,
            device_placement=True,
        )

        # Initialize tokenizer with safety settings
        tokenizer = AutoTokenizer.from_pretrained(
            'gpt2',
            model_max_length=config['model']['max_position_embeddings'],
            padding_side='right',
            truncation_side='right',
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Update config with actual vocab size and add safety margin
        vocab_size = len(tokenizer)
        config['model']['vocab_size'] = vocab_size + 1  # Add 1 for safety
        logger.info(f"Using vocabulary size: {config['model']['vocab_size']} (original: {vocab_size})")
        
        # Verify token indices will be valid
        if tokenizer.vocab_size > config['model']['vocab_size']:
            raise ValueError(
                f"Tokenizer vocabulary size ({tokenizer.vocab_size}) is larger than "
                f"model vocabulary size ({config['model']['vocab_size']})"
            )

        # Create dataset and dataloader with safety checks
        try:
            dataset = TextDataset(args.input_file, tokenizer)
            # Verify dataset token indices
            max_token_id = max(max(x) for x in dataset.examples)
            if max_token_id >= config['model']['vocab_size']:
                raise ValueError(
                    f"Dataset contains token ID {max_token_id} which is >= vocabulary size "
                    f"{config['model']['vocab_size']}"
                )
            logger.info(f"Dataset token ID range: 0 to {max_token_id}")
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            raise

        dataloader = create_dataloader(
            dataset, 
            config['training']['batch_size'],
            num_workers=0  # Disable multiprocessing for tokenizer
        )

        # Initialize model with updated config
        model_config = SmolLM2Config(**config['model'])
        model = SmolLM2(model_config)
        
        # Add embedding size check
        embed_size = model.embed_tokens.weight.size(0)
        if embed_size != config['model']['vocab_size']:
            raise ValueError(
                f"Model embedding size ({embed_size}) doesn't match "
                f"configured vocabulary size ({config['model']['vocab_size']})"
            )
        
        # Move model to device before creating optimizer
        model = model.to(device)
        model = accelerator.prepare_model(model)
        
        optimizer = AdamW(
            model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            eps=1e-8,  # Added for stability
            betas=(0.9, 0.999),  # Added explicit betas
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
        optimizer, dataloader, scheduler = accelerator.prepare(
            optimizer, dataloader, scheduler
        )

        # Training loop
        model.train()
        data_iter = iter(dataloader)
        total_steps = start_step + args.train_steps
        step = start_step
        last_save_successful = True

        # Enable tensor cores for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        while step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device and ensure it's contiguous
            batch = batch.to(model.device, non_blocking=True).contiguous()

            with accelerator.accumulate(model):
                # Forward pass with gradient checkpointing
                outputs, loss = model(batch, labels=batch)
                
                # Scale loss for gradient accumulation
                loss = loss / config['training']['gradient_accumulation_steps']
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    # Clip gradients
                    accelerator.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                    
                    # Step optimizer and scheduler
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            if step % config['training']['eval_steps'] == 0:
                # Gather loss from all processes
                avg_loss = accelerator.gather(loss).mean().item()
                avg_loss *= config['training']['gradient_accumulation_steps']  # Rescale loss
                logger.info(f"Step {step}: loss = {avg_loss:.4f}")
                if os.environ.get("WANDB_DISABLED", "").lower() != "true":
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "step": step,
                    })

            if step % config['training']['prediction_steps'] == 0:
                model.eval()  # Set to eval mode for generation
                sample = generate_sample(model, tokenizer)
                model.train()  # Set back to training mode
                logger.info(f"\nGenerated sample at step {step}:\n{sample}\n")
                if os.environ.get("WANDB_DISABLED", "").lower() != "true":
                    wandb.log({"generated_text": sample, "step": step})

            if step % config['training']['save_steps'] == 0 or step == total_steps - 1:
                # Unwrap and save model
                unwrapped_model = accelerator.unwrap_model(model)
                last_save_successful = save_checkpoint(
                    unwrapped_model,
                    optimizer,
                    scheduler,
                    step + 1,  # Save with the next step number
                    loss.item() * config['training']['gradient_accumulation_steps'],  # Save unscaled loss
                    args.save_dir,
                )
                if not last_save_successful:
                    logger.warning("Failed to save checkpoint, but continuing training...")

            step += 1

    except Exception as e:
        logger.error(f"Training interrupted at step {step}: {str(e)}")
        if 'last_save_successful' in locals() and not last_save_successful:
            try:
                save_checkpoint(
                    accelerator.unwrap_model(model),
                    optimizer,
                    scheduler,
                    step,
                    loss.item() * config['training']['gradient_accumulation_steps'],
                    args.save_dir,
                )
            except Exception as save_error:
                logger.error(f"Failed to save final checkpoint: {str(save_error)}")
        raise

    finally:
        if os.environ.get("WANDB_DISABLED", "").lower() != "true":
            wandb.finish()
        accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()

    # Set WandB disabled environment variable if specified
    if args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    try:
        train(args)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise 