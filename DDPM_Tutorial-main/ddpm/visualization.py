"""
Visualization helpers for the DDPM Swiss Roll tutorial.

Provides functions to plot:
  - The original Swiss Roll distribution
  - Forward diffusion snapshots at selected timesteps
  - Reverse denoising trajectory
  - Training loss curve
"""

# Store annotations for later evaluation, including references such as plt.Axes.
from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================

# Accept either strings or Path objects for optional figure destinations.
from pathlib import Path
# Describe lists of snapshots and sequences of selected indices in the type hints.
from typing import List, Sequence

# Use Matplotlib to create figures, plotting areas, and scatter plots.
import matplotlib.pyplot as plt
# Use NumPy arrays and evenly spaced snapshot indices.
import numpy as np
# Recognize PyTorch tensor inputs before converting their coordinates for plotting.
import torch


# =============================================================================
# Shared Scatter-Plot Styling
# =============================================================================

def _scatter(
    ax: plt.Axes,
    points: np.ndarray,
    title: str | None = None,
    alpha: float = 0.4,
    s: float = 8,
    color: str = "#4C72B0",
) -> None:
    """Internal helper: draw a styled scatter plot on *ax*."""
    # Draw one marker for each row of the (N, 2) coordinate array.
    ## Column 0 supplies horizontal positions; column 1 supplies vertical positions.
    ## alpha controls transparency, s controls marker size, and color sets the fill.
    ax.scatter(points[:, 0], points[:, 1], alpha=alpha, s=s, edgecolors="white",
               linewidths=0.3, color=color)
    # Add a title only when the caller provides a nonempty value.
    if title:
        ax.set_title(title, fontsize=11)
    # Keep one horizontal unit the same visual length as one vertical unit.
    ## A faint grid helps compare the coordinate spread without obscuring the points.
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


# =============================================================================
# Original Swiss Roll Distribution
# =============================================================================

def plot_swiss_roll(
    data: np.ndarray | torch.Tensor,
    save_path: str | Path | None = None,
) -> None:
    """Plot the original 2D Swiss Roll distribution."""
    # Convert a tensor to the NumPy representation expected by the plotting helper.
    ## Direct .numpy() requires a CPU tensor with no gradient tracking.
    ## These helpers expect plotting-ready tensors; they do not move GPU data to CPU.
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    # Create one plotting area in a 6-by-6-inch figure and draw the original points.
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter(ax, data, title="Swiss Roll Distribution")
    # Adjust the figure spacing so labels and titles fit inside its boundaries.
    fig.tight_layout()
    # Optionally save a 150-dpi image, trimming excess whitespace around the figure.
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    # Display the figure, including inline output when called from the notebook.
    plt.show()


# =============================================================================
# Forward Diffusion Snapshots
# =============================================================================

def plot_forward_process(
    snapshots: List[torch.Tensor],
    timesteps: Sequence[int],
    save_path: str | Path | None = None,
) -> None:
    """
    Visualise the forward diffusion at selected timesteps.

    Args:
        snapshots: List of (N, 2) tensors for each plotted timestep.
        timesteps: Corresponding timestep labels.
    """
    # Allocate one plotting area per supplied noisy snapshot.
    ## Each snapshot contains N points with two coordinates per point.
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    # With one subplot, Matplotlib returns a single Axes object instead of an array.
    ## Wrapping it in a list lets the same loop handle both cases.
    if n == 1:
        axes = [axes]

    # Pair each plotting area with a snapshot and its matching timestep label.
    ## Snapshots sampled independently with q_sample show forward marginals;
    ## they need not describe one realized step-by-step random trajectory.
    for ax, snap, t in zip(axes, snapshots, timesteps):
        # Convert CPU tensors without gradient tracking; leave NumPy arrays as they are.
        pts = snap.numpy() if isinstance(snap, torch.Tensor) else snap
        # Draw the point cloud and label the supplied schedule index.
        _scatter(ax, pts, title=f"t = {t}")

    # Add a shared title above the subplot titles and arrange the panel spacing.
    fig.suptitle("Forward Diffusion Process  q(x_t | x_0)", fontsize=13, y=1.02)
    fig.tight_layout()
    # Save only when a destination was supplied, then display the complete figure.
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Reverse Denoising Trajectory
# =============================================================================

def plot_reverse_process(
    trajectory: List[torch.Tensor],
    steps_to_show: Sequence[int] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Visualise the reverse denoising trajectory.

    Args:
        trajectory: Full list from p_sample_loop (x_T, ..., x_0).
        steps_to_show: Indices into *trajectory* to display.
                        If None, ~6 evenly spaced snapshots are picked.
    """
    # Count stored states, including both the initial noise and the final samples.
    ## A full trajectory with T reverse updates has T + 1 entries.
    total = len(trajectory)
    # By default, choose at most six approximately evenly spaced trajectory indices.
    ## Include index 0 and index total - 1 so the starting and final states are shown.
    if steps_to_show is None:
        steps_to_show = np.linspace(0, total - 1, min(6, total), dtype=int)

    # Create one subplot for each selected trajectory entry.
    n = len(steps_to_show)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    # Normalize the single-subplot case to the same iterable form as multiple panels.
    if n == 1:
        axes = [axes]

    # Visit the saved states in the order specified by steps_to_show.
    for ax, idx in zip(axes, steps_to_show):
        # Read one (N, 2) snapshot and convert it to a NumPy array when needed.
        ## p_sample_loop stores its trajectory tensors on CPU for this plotting step.
        pts = trajectory[idx]
        if isinstance(pts, torch.Tensor):
            pts = pts.numpy()
        # Label the number of reverse updates remaining after this saved state.
        ## For 101 states: index 0 means 100 updates remain; index 100 means zero.
        ### A trajectory index counts saved states, not the schedule index used by p_sample.
        remaining = total - 1 - idx
        _scatter(ax, pts, title=f"reverse step {remaining}")

    # Give the panels a shared title and adjust their spacing.
    fig.suptitle("Reverse Denoising Process  p(x_{{t-1}} | x_t)", fontsize=13, y=1.02)
    fig.tight_layout()
    # Optionally write the figure to disk, then display the denoising snapshots.
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# Training Loss History
# =============================================================================

def plot_training_loss(
    losses: List[float],
    save_path: str | Path | None = None,
) -> None:
    """Plot the training loss curve."""
    # Create one plotting area for the recorded average loss of each epoch.
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot the values in order; implicit horizontal indices run from 0 to len(losses) - 1.
    ## Training draws fresh noise and timesteps, so the curve can fluctuate between epochs.
    ax.plot(losses, linewidth=0.8, color="#4C72B0")
    # Label the training progress, MSE values, and plot title, then add a faint grid.
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.2)
    # Fit labels within the figure, optionally save the image, and display the curve.
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
