"""
Swiss Roll dataset utilities for the DDPM tutorial.

The Swiss Roll is a classic 2D manifold that serves as an excellent toy
distribution for testing generative models.  Its spiral structure is
non-trivial enough to verify that the diffusion model has genuinely
learned the data distribution rather than merely memorising noise.
"""

# =============================================================================
# Imports
# =============================================================================

# Use PyTorch tensors for model inputs and random mini-batch indices.
import torch
# Use NumPy's array type in the plotting interface's return annotation.
import numpy as np
# Generate a three-dimensional Swiss Roll before selecting its spiral plane.
from sklearn.datasets import make_swiss_roll


# =============================================================================
# Generate and Store the Training Dataset
# =============================================================================

class SwissRollDataset:
    """
    Generates and wraps a 2D Swiss Roll point cloud.

    The 3D Swiss Roll from sklearn is projected onto the (x, z) plane
    and divided by a fixed scale to reduce the coordinate magnitudes.

    Args:
        n_samples: Number of data points.
        noise: Standard deviation of Gaussian noise added to the roll.
        scale: Divisor applied after projection (controls spread).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        noise: float = 1.0,
        scale: float = 10.0,
        seed: int = 42,
    ):
        # Keep the dataset settings available as attributes of this object.
        ## self refers to this particular SwissRollDataset instance.
        self.n_samples = n_samples
        self.noise = noise
        self.scale = scale

        # Generate n_samples three-dimensional points, with shape (n_samples, 3).
        ## random_state makes dataset generation reproducible for the same seed.
        ## The second return value contains positions along the roll; _ discards it.
        points_3d, _ = make_swiss_roll(n_samples, noise=noise, random_state=seed)
        # Project to 2D (x, z) and rescale
        ## [:, [0, 2]] keeps every row and selects the first and third coordinates.
        ## Dividing by scale changes the coordinate units; it does not subtract the
        ## mean or divide by a measured standard deviation.
        ### The resulting CPU tensor has shape (n_samples, 2), using float32 values.
        self.data = torch.tensor(
            points_3d[:, [0, 2]] / scale,
            dtype=torch.float32,
        )

    def get_tensor(self) -> torch.Tensor:
        """Return the full dataset as a (n_samples, 2) tensor."""
        # Return the stored tensor directly so training code can select batches.
        return self.data

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        """Sample a random mini-batch from the dataset."""
        # Draw batch_size indices independently from 0 through n_samples - 1.
        ## Sampling is with replacement: the same point can appear more than once.
        ## This draw uses PyTorch's random generator, separate from random_state above.
        indices = torch.randint(0, self.n_samples, (batch_size,))
        # Select the corresponding rows, producing a tensor of shape (batch_size, 2).
        return self.data[indices]

    def get_numpy(self) -> np.ndarray:
        """Return data as a NumPy array for plotting."""
        # Expose the stored CPU coordinates as a NumPy array of shape (n_samples, 2).
        ## This conversion can share memory with self.data rather than copy its values.
        return self.data.numpy()
