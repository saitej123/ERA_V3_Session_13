import os
import torch
import wandb
import argparse
from train import train

def parse_continue_args():
    parser = argparse.ArgumentParser(description='Continue training SmolLM2')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input text file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--train_steps', type=int, required=True, help='Number of training steps')
    parser.add_argument('--start_step', type=int, default=5000, help='Starting step for continued training')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_continue_args()
    
    # Set checkpoint path
    args.checkpoint_path = os.path.join(args.save_dir, f"step_{args.start_step}.pt")
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    print(f"Loading checkpoint from {args.checkpoint_path}")
    
    # Continue training without reinitializing wandb
    train(args)

if __name__ == "__main__":
    main() 