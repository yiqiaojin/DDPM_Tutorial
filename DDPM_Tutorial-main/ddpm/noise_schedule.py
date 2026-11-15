"""
Noise schedule for DDPM.

Defines beta (variance) schedules and computes derived quantities
(alpha, alpha_bar) used throughout the diffusion process.
"""

# =============================================================================
# Import
# =============================================================================

# Load PyTorch for creating schedule tensors and computing their cumulative products.
import torch


# =============================================================================
# Noise Schedule: How Much Noise to Add at Each Transition
# =============================================================================

# Keep the fixed schedule quantities together so forward and reverse diffusion
# can use exactly the same beta, alpha, and cumulative-alpha values.
class NoiseSchedule:
    """
    Manages the noise schedule for a diffusion process.

    The schedule controls how much noise is added at each timestep.
    Supports linear and sigmoid schedule types.

    Attributes:
        n_steps: Total number of diffusion steps.
        betas: Variance schedule at each timestep, shape (n_steps,).
        alphas: 1 - betas, shape (n_steps,).
        alphas_bar: Cumulative product of alphas, shape (n_steps,).
    """

    # Construct one scalar coefficient per diffusion transition.
    ## n_steps controls the schedule length; beta_start and beta_end set its scale.
    def __init__(
        self,
        n_steps: int = 100,
        beta_start: float = 1e-5,
        beta_end: float = 1e-2,
        schedule_type: str = "sigmoid",
    ):
        # Store the number of transitions for training-time index sampling and generation.
        ## Code indices run from 0 to n_steps - 1; index 0 is the first noising transition.
        ### The clean starting point is x0, before any entry of this schedule is applied.
        self.n_steps = n_steps

        # Build the requested variance schedule, with shape (n_steps,).
        ## Each beta is the variance of the independent Gaussian noise added in one step.
        self.betas = self._create_schedule(n_steps, beta_start, beta_end, schedule_type)

        # Compute the squared signal-scaling coefficient for each individual transition.
        ## The forward update scales the previous state by sqrt(alpha_t).
        self.alphas = 1.0 - self.betas

        # Multiply alpha values cumulatively along the one-dimensional schedule.
        ## alphas_bar[t] = alphas[0] * ... * alphas[t], with shape (n_steps,).
        ### sqrt(alphas_bar[t]) scales clean x0 when sampling a noisy point directly.
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

        # With the default 100-step schedule, alphas_bar[-1] is about 0.605.
        ## The last forward state therefore retains substantial signal from x0.
        ## Starting reverse generation from a standard Gaussian is an approximation
        ## for these defaults, rather than an exact match to the terminal forward state.

    # A static method uses its arguments without needing an existing schedule instance.
    @staticmethod
    def _create_schedule(
        n_steps: int,
        beta_start: float,
        beta_end: float,
        schedule_type: str,
    ) -> torch.Tensor:
        """
        Create the beta schedule.

        Args:
            n_steps: Number of diffusion steps.
            beta_start: Minimum beta value.
            beta_end: Maximum beta value.
            schedule_type: "sigmoid" or "linear".

        Returns:
            Beta values of shape (n_steps,).
        """
        # Select the formula requested by the caller.
        if schedule_type == "sigmoid":
            # Create n_steps evenly spaced inputs from -6 to 6 and apply sigmoid(u).
            ## sigmoid(u) = 1 / (1 + exp(-u)); its output grows smoothly toward 1.
            betas = torch.sigmoid(torch.linspace(-6, 6, n_steps))

            # Rescale sigmoid values into the interval bounded by beta_start and beta_end.
            ## Finite inputs -6 and 6 approach these bounds without reaching them exactly.
            betas = betas * (beta_end - beta_start) + beta_start
        elif schedule_type == "linear":
            # Increase beta by a constant amount between the two endpoint values.
            betas = torch.linspace(beta_start, beta_end, n_steps)
        else:
            # Report unsupported names so a spelling mistake does not select another formula.
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Return the one-dimensional tensor used to derive all remaining coefficients.
        return betas

    # -------------------------------------------------------------------------
    # Move Schedule Tensors to the Computation Device
    # -------------------------------------------------------------------------

    def to(self, device: torch.device) -> "NoiseSchedule":
        """Move all tensors to the specified device."""
        # Replace each tensor with its copy on the requested CPU or CUDA device.
        ## Schedule tensors must be on a compatible device with the data and model.
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)

        # Return this same schedule object to allow calls such as NoiseSchedule().to(device).
        return self
