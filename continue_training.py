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

def setup_wandb(args):
    """Initialize WandB with proper configuration."""
    run = wandb.init(
        project="smollm2-training",
        name=f"smollm2-continued-training-{args.start_step}",
        config={
            "start_step": args.start_step,
            "train_steps": args.train_steps,
            "input_file": args.input_file,
            "save_dir": args.save_dir,
            "config_file": args.config,
            "phase": "continuation"
        },
        resume="allow"
    )
    return run

def main():
    # Parse arguments
    args = parse_continue_args()
    
    # Set checkpoint path
    args.checkpoint_path = os.path.join(args.save_dir, f"step_{args.start_step}.pt")
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    print(f"Loading checkpoint from {args.checkpoint_path}")
    
    try:
        # Initialize WandB
        run = setup_wandb(args)
        
        # Log start of continued training
        wandb.log({
            "continued_training_start": True,
            "starting_step": args.start_step,
        }, step=args.start_step)
        
        # Continue training
        train(args)
        
        # Log completion
        wandb.log({
            "continued_training_complete": True,
            "final_step": args.start_step + args.train_steps
        }, step=args.start_step + args.train_steps)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Ensure WandB is properly finished
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 