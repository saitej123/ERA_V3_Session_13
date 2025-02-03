import argparse
import os
import yaml
import math
import logging
from pathlib import Path
from typing import Optional

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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    loss: float,
    save_dir: str,
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'loss': loss,
    }
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, f'step_{step}.pt'))
    logger.info(f"Saved checkpoint at step {step}")


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
    model_config = SmolLM2Config(**config['model'])

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

        if step % config['training']['save_steps'] == 0:
            save_checkpoint(
                accelerator.unwrap_model(model),
                optimizer,
                scheduler,
                step,
                loss.item(),
                args.save_dir,
            )

        step += 1

    # Save final checkpoint
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

    train(args) 