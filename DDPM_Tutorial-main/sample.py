"""
Sampling script — generate new points from a trained DDPM model.

Usage:
    python sample.py                            # defaults
    python sample.py --n_samples 5000           # more points
    python sample.py --checkpoint outputs/model.pt
"""

# =============================================================================
# Imports
# =============================================================================

# Read sampling options supplied on the command line.
import argparse
# Build paths to the checkpoints and output figures.
from pathlib import Path

# Load the saved PyTorch parameters and run the reverse diffusion process.
import torch
# Draw the comparison between the reference and generated point clouds.
import matplotlib.pyplot as plt

# Import the reusable model, diffusion, dataset, and plotting components.
from ddpm import (
    NoiseSchedule,
    ConditionalModel,
    GaussianDiffusion,
    SwissRollDataset,
    plot_reverse_process,
)


# =============================================================================
# Command-Line Arguments
# =============================================================================

# Store the requested sampling settings as attributes of an argparse.Namespace.
def parse_args() -> argparse.Namespace:
    # Describe the script in its --help output.
    p = argparse.ArgumentParser(description="Sample from trained DDPM")
    # Select the learned parameter tensors written by train.py.
    p.add_argument("--checkpoint", type=str, default="outputs/model.pt")
    # Select the matching noise schedule settings saved alongside that checkpoint.
    p.add_argument("--config", type=str, default="outputs/config.pt")
    # Request this many independently initialized points, each with two coordinates.
    p.add_argument("--n_samples", type=int, default=10000)
    # Choose CPU or an available CUDA device supported by the installed PyTorch build.
    p.add_argument("--device", type=str, default="cpu")
    # Use the output folder prepared during training, or another existing folder.
    ## Relative paths are resolved from the terminal's current working directory.
    p.add_argument("--save_dir", type=str, default="outputs")
    # Seed the PyTorch random draws used during reverse sampling.
    p.add_argument("--seed", type=int, default=0)
    # Fill in defaults for omitted options and return the parsed settings.
    return p.parse_args()


# =============================================================================
# Load the Trained Model and Generate Points
# =============================================================================

# Reconstruct the trained diffusion setup, generate a reverse trajectory, and plot it.
def sample(args: argparse.Namespace) -> None:
    # Seed PyTorch's random generator before model construction and Gaussian sampling.
    torch.manual_seed(args.seed)
    # Represent the chosen output folder as a Path object for joining filenames.
    save_dir = Path(args.save_dir)
    # Keep the model, schedule tensors, and reverse updates on the same device.
    device = torch.device(args.device)

    # =========================================================================
    # Reconstruct the Training Schedule and Load the Learned Parameters
    # =========================================================================

    # Read the saved dictionary of primitive schedule settings.
    ## weights_only=True restricts deserialization to supported weights/basic values.
    ## map_location controls the device used for any tensors in the saved object.
    cfg = torch.load(args.config, map_location=device, weights_only=True)
    # Recreate exactly the beta schedule used to train this predictor.
    schedule = NoiseSchedule(
        n_steps=cfg["n_steps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        schedule_type=cfg["schedule"],
    ).to(device)

    # Rebuild the MLP with the same number of timestep embedding entries as training.
    model = ConditionalModel(n_steps=cfg["n_steps"]).to(device)
    # Match each saved parameter tensor to its corresponding layer in the new model.
    ## The checkpoint's parameter names and shapes must match this architecture.
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    # Set evaluation behavior before sampling.
    ## The diffusion sampling methods separately disable gradient recording.
    model.eval()

    # Use the loaded schedule for all reverse mean and noise calculations.
    diffusion = GaussianDiffusion(schedule)

    # =========================================================================
    # Run the Full Reverse Denoising Chain
    # =========================================================================

    # Report the requested number of points and the number of reverse updates.
    print(f"Sampling {args.n_samples} points ({cfg['n_steps']} reverse steps) ...")
    # Begin with standard Gaussian points and apply the learned updates in reverse order.
    ## Shape (N, 2) requests N two-dimensional points.
    ## return_trajectory=True retains n_steps + 1 states, including the starting noise.
    trajectory = diffusion.p_sample_loop(
        model,
        shape=(args.n_samples, 2),
        device=device,
        return_trajectory=True,
    )
    # The last saved state contains the final generated coordinates, with shape (N, 2).
    ## p_sample_loop stores trajectory tensors on CPU, even for CUDA reverse updates.
    generated = trajectory[-1]
    print("Done.")

    # =========================================================================
    # Visualise the Reverse Chain and the Generated Distribution
    # =========================================================================

    # Save and display selected snapshots of the noise-to-data trajectory.
    plot_reverse_process(trajectory, save_path=save_dir / "reverse_process.png")

    # Draw a reference Swiss Roll cloud with the same number of points.
    ## This dataset uses its default seed; compare the clouds' overall shape and density.
    real = SwissRollDataset(n_samples=args.n_samples).get_numpy()
    # Convert the final CPU tensor to a NumPy array for Matplotlib.
    gen_np = generated.numpy()

    # Create two plotting areas, one for each distribution.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Unpack a plotting area, a point array, and a title for each panel.
    for ax, pts, label in [
        (axes[0], real, "Real Swiss Roll"),
        (axes[1], gen_np, "Generated (DDPM)"),
    ]:
        # Plot the first coordinate horizontally and the second vertically.
        ## alpha sets transparency; s sets marker size; white edges separate nearby markers.
        ax.scatter(pts[:, 0], pts[:, 1], alpha=0.4, s=8,
                   edgecolors="white", linewidths=0.3)
        # Label each distribution and display equal horizontal and vertical coordinate units.
        ax.set_title(label, fontsize=12)
        ax.set_aspect("equal")
        # Add a faint coordinate grid to each panel.
        ax.grid(True, alpha=0.2)

    # Add a shared title and adjust spacing so subplot labels fit.
    fig.suptitle("Real vs. Generated Distribution", fontsize=14)
    fig.tight_layout()
    # Save the figure at 150 dots per inch, including its labels in the bounding box.
    fig.savefig(save_dir / "comparison.png", dpi=150, bbox_inches="tight")
    # Display the comparison and report the destination of the saved figures.
    plt.show()
    print(f"Figures saved to {save_dir}/")


# =============================================================================
# Script Entry Point
# =============================================================================

# Running python sample.py parses its options and begins generation.
## The guard lets other Python code import these functions for reuse.
if __name__ == "__main__":
    sample(parse_args())
