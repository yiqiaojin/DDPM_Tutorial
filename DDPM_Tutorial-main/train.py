"""
Training script for the DDPM Swiss Roll tutorial.

Usage:
    python train.py                     # train with defaults
    python train.py --epochs 2000       # more epochs
    python train.py --device cuda       # use GPU
    python train.py --schedule linear   # linear beta schedule
"""

# =============================================================================
# Imports
# =============================================================================

# Read command-line options such as --epochs 2000.
import argparse
# Represent output directories and file paths using Path objects.
from pathlib import Path

# Load PyTorch for tensors, model training, and checkpoint files.
import torch
# Display a progress bar in the terminal during training.
from tqdm import tqdm

# Import the reusable components exposed by ddpm/__init__.py.
from ddpm import (
    NoiseSchedule,
    ConditionalModel,
    GaussianDiffusion,
    SwissRollDataset,
    plot_training_loss,
    plot_swiss_roll,
)


# =============================================================================
# Command-Line Arguments
# =============================================================================

# Collect settings supplied after python train.py into an argparse.Namespace.
## Each option becomes an attribute, for example --epochs becomes args.epochs.
def parse_args() -> argparse.Namespace:
    # Create the parser and the description shown by python train.py --help.
    p = argparse.ArgumentParser(description="Train DDPM on 2D Swiss Roll")
    # Set the length of the forward schedule and the reverse generation chain.
    p.add_argument("--n_steps", type=int, default=100, help="Number of diffusion steps")
    # An epoch is one complete pass through the fixed training dataset.
    p.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    # Process at most this many points in each optimizer update.
    p.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    # Set Adam's learning rate: 1e-3 means 0.001.
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    # Generate this many Swiss Roll points before training begins.
    p.add_argument("--n_samples", type=int, default=10000, help="Dataset size")
    # Set the bounds used to construct the per-transition noise variances.
    p.add_argument("--beta_start", type=float, default=1e-5, help="Min beta")
    p.add_argument("--beta_end", type=float, default=1e-2, help="Max beta")
    # Restrict the schedule choice to the two implementations supported by the library.
    p.add_argument("--schedule", type=str, default="sigmoid", choices=["sigmoid", "linear"])
    # Choose CPU or a CUDA device supported by the installed PyTorch build and hardware.
    p.add_argument("--device", type=str, default="cpu")
    # Relative output paths are resolved from the terminal's current working directory.
    p.add_argument("--save_dir", type=str, default="outputs", help="Output directory")
    # Use this seed for PyTorch random draws and the separately seeded dataset generator.
    p.add_argument("--seed", type=int, default=42)
    # Parse the supplied options, filling in defaults for omitted arguments.
    return p.parse_args()


# =============================================================================
# Training Workflow
# =============================================================================

# Build the data, schedule, and model; train the predictor; then save its artifacts.
def train(args: argparse.Namespace) -> None:
    # Seed PyTorch's random draws, including model initialization and batch shuffling.
    torch.manual_seed(args.seed)
    # Create the output folder and any missing parent folders.
    ## exist_ok=True also permits using an output folder that already exists.
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Use the same device for the data, model parameters, and schedule tensors.
    device = torch.device(args.device)

    # =========================================================================
    # Generate and Inspect the Training Data
    # =========================================================================

    # Generate a fixed point cloud using the supplied dataset size and random seed.
    dataset = SwissRollDataset(n_samples=args.n_samples, seed=args.seed)
    # Move the coordinates to the training device; data has shape (N, 2).
    data = dataset.get_tensor().to(device)

    # Report the number of points N and the coordinate dimension, which is 2.
    print(f"Dataset: {data.shape[0]} points, dim={data.shape[1]}")
    # Save and display the original distribution using the dataset's CPU NumPy array.
    plot_swiss_roll(dataset.get_numpy(), save_path=save_dir / "swiss_roll.png")

    # =========================================================================
    # Set Up the Noise Schedule and Diffusion Equations
    # =========================================================================

    # Precompute beta, alpha, and cumulative alpha from the command-line settings.
    ## Schedule indices run from 0 to n_steps - 1; index 0 is the first noising step.
    schedule = NoiseSchedule(
        n_steps=args.n_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule_type=args.schedule,
    ).to(device)
    # Give the forward sampler, loss, and reverse sampler access to that same schedule.
    diffusion = GaussianDiffusion(schedule)

    # =========================================================================
    # Initialize the Noise Predictor and Optimizer
    # =========================================================================

    # Create the MLP, including a learned embedding entry for every timestep.
    ## A newly constructed PyTorch model starts in training mode.
    model = ConditionalModel(n_steps=args.n_steps).to(device)
    # Let Adam update the linear weights, biases, and timestep embeddings.
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    # numel() counts scalar entries in each parameter tensor; sum gives their total.
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # =========================================================================
    # Train over Shuffled Mini-Batches
    # =========================================================================

    # Retain one average loss per epoch for the final training-curve plot.
    epoch_losses = []
    # Wrap the epoch indices in a progress bar that can also display the current loss.
    pbar = tqdm(range(args.epochs), desc="Training")

    for epoch in pbar:
        # Randomly order all N point indices once per epoch, on the training device.
        permutation = torch.randperm(data.shape[0], device=device)
        # Collect ordinary Python numbers rather than tensors holding computation graphs.
        batch_losses = []

        # Visit the shuffled indices in chunks; the final chunk can be smaller.
        for i in range(0, data.shape[0], args.batch_size):
            # Select B points, producing a mini-batch with shape (B, 2).
            batch = data[permutation[i : i + args.batch_size]]
            # Sample fresh timesteps and noise, then compare predicted noise with true noise.
            loss = diffusion.noise_estimation_loss(model, batch)

            # Clear the gradients accumulated during the preceding update.
            optimiser.zero_grad()
            # Differentiate the scalar loss with respect to the model's parameters.
            loss.backward()
            # Rescale gradients if their combined L2 norm exceeds 1.0.
            ## Gradient clipping limits unusually large updates before Adam uses them.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update the trainable parameters using Adam and the current gradients.
            optimiser.step()
            # Extract the scalar loss value for logging without retaining its graph.
            batch_losses.append(loss.item())

        # Average the batch means for monitoring this epoch.
        ## Each batch receives equal weight, including a potentially smaller final batch.
        avg_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(avg_loss)
        # Show this epoch's average next to the terminal progress bar.
        pbar.set_postfix(loss=f"{avg_loss:.6f}")

    # =========================================================================
    # Save the Trained Parameters, Schedule Settings, and Loss Curve
    # =========================================================================

    # A state_dict maps layer names to their learned parameter tensors.
    ## sample.py creates the matching architecture and loads these values into it.
    torch.save(model.state_dict(), save_dir / "model.pt")
    # Save the schedule settings required to reproduce the training noise process.
    ## n_steps also determines the size of the model's timestep embedding tables.
    torch.save(
        {
            "n_steps": args.n_steps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "schedule": args.schedule,
        },
        save_dir / "config.pt",
    )
    # Save and display the recorded average loss as a function of the training epoch.
    plot_training_loss(epoch_losses, save_path=save_dir / "loss.png")
    print(f"Model & loss curve saved to {save_dir}/")


# =============================================================================
# Script Entry Point
# =============================================================================

# Running python train.py parses the options and starts the workflow above.
## Importing this file as a module leaves the functions available for callers.
if __name__ == "__main__":
    train(parse_args())
