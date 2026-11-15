# =============================================================================
# Public Package Interface
# =============================================================================

# A leading dot imports from another module inside this same ddpm package.
## Re-exporting these names allows callers to write from ddpm import NoiseSchedule.
# Import the schedule that stores each transition's noise and signal coefficients.
from .noise_schedule import NoiseSchedule
# Import both the conditioned layer and the complete noise-prediction network.
from .model import ConditionalLinear, ConditionalModel
# Import the forward noising, training-loss, and reverse-sampling interface.
from .diffusion import GaussianDiffusion
# Import the class that generates and stores the 2D training coordinates.
from .dataset import SwissRollDataset
# Make the plotting helpers available through the same package-level interface.
from .visualization import (
    plot_swiss_roll,
    plot_forward_process,
    plot_reverse_process,
    plot_training_loss,
)

# Declare the intended public names, including those used by from ddpm import *.
## These strings name the objects imported above; they do not create new copies.
__all__ = [
    "NoiseSchedule",
    "ConditionalLinear",
    "ConditionalModel",
    "GaussianDiffusion",
    "SwissRollDataset",
    "plot_swiss_roll",
    "plot_forward_process",
    "plot_reverse_process",
    "plot_training_loss",
]
