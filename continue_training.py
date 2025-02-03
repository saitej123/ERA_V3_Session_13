import os
import torch
import wandb
from train import train, parse_args

def main():
    # Parse arguments
    args = parse_args()
    
    # Override some arguments for continued training
    args.checkpoint_path = "checkpoints/step_5000.pt"
    args.train_steps = 50  # Train for 50 more steps
    args.start_step = 5000  # Start from step 5000
    
    # Initialize wandb for continued training
    wandb.init(
        project="smollm2-training",
        name="smollm2-continued-training",
        config=args,
        resume="allow"
    )
    
    # Load checkpoint and continue training
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path)
    
    # Log that we're continuing from step 5000
    wandb.log({
        "continued_training_start": True,
        "starting_step": 5000,
        "previous_loss": checkpoint["loss"]
    })
    
    # Continue training
    train(args, checkpoint)
    
    # Log completion
    wandb.log({
        "continued_training_complete": True,
        "final_step": 5050
    })
    
    wandb.finish()

if __name__ == "__main__":
    main() 