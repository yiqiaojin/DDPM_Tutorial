"""
Gaussian diffusion process — forward noising and reverse denoising.

Implements the core DDPM algorithm from:
    Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.

Key formulas
------------
Forward (closed-form):
    x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps

Reverse (single step):
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps_theta)
              + sigma_t * z

Training loss (simplified):
    L = E_{t, x_0, eps} || eps - eps_theta(x_t, t) ||^2
"""

# =============================================================================
# Imports and Type Annotations
# =============================================================================

# Postpone evaluation of type annotations until they are explicitly inspected.
## This lets the class names below appear in hints without runtime imports.
from __future__ import annotations

# TYPE_CHECKING is true for static type checkers and false when this module runs.
## List describes the optional list of tensors returned by the sampling loop.
from typing import TYPE_CHECKING, List

# Load tensor operations, random sampling, and the no_grad decorator.
import torch

# Give type-checking tools access to the model and schedule class definitions.
## These imports are skipped at runtime; the postponed hints above permit this.
if TYPE_CHECKING:
    from .model import ConditionalModel
    from .noise_schedule import NoiseSchedule


# =============================================================================
# Forward Diffusion, Training Loss, and Reverse Sampling
# =============================================================================

# Group the fixed noising equations and learned reverse sampling in one object.
## The model owns trainable parameters; this helper uses a supplied noise schedule.
class GaussianDiffusion:
    """
    Encapsulates the forward and reverse diffusion processes.

    Args:
        schedule: A NoiseSchedule instance with precomputed betas/alphas.
    """

    def __init__(self, schedule: NoiseSchedule):
        # Keep the schedule object so every method uses the same betas and alphas.
        ## Its tensors should be on the same device as the data and the model.
        self.schedule = schedule

    # =============================================================================
    # Forward process  q(x_t | x_0)
    # =============================================================================

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) using the reparameterisation trick.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0: Clean data, shape (B, D).
            t: Timestep indices, shape (B,).
            noise: Optional pre-sampled noise, shape (B, D).

        Returns:
            Noisy samples x_t, shape (B, D).
        """
        # Draw noise only when the caller has not supplied a particular realization.
        ## Supplying noise explicitly lets training reuse it as the prediction target.
        if noise is None:
            # Match x0's shape (B, D), dtype, and device; entries follow N(0, 1).
            noise = torch.randn_like(x0)

        # Select one cumulative alpha value per point using t of shape (B,).
        ## unsqueeze(1) changes the indexed result from (B,) to (B, 1).
        ### Broadcasting then applies each point's coefficient to all D coordinates.
        ### Schedule index 0 includes the first noising transition; it is not clean x0.
        alpha_bar_t = self.schedule.alphas_bar[t].unsqueeze(1)  # (B, 1)
        # Combine scaled clean coordinates with scaled Gaussian noise, giving (B, D).
        ## sqrt(alpha_bar_t) scales signal; sqrt(1 - alpha_bar_t) scales noise.
        ### This closed form skips the intermediate forward transitions.
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    # =============================================================================
    # Training loss
    # =============================================================================

    def noise_estimation_loss(
        self,
        model: ConditionalModel,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the simplified DDPM loss:
            L = || eps - eps_theta(x_t, t) ||^2

        A random timestep t is sampled *per element* in the batch.

        Args:
            model: Network that predicts eps_theta(x_t, t).
            x0: Clean data batch, shape (B, D).

        Returns:
            Scalar MSE loss.
        """
        # Read B, the number of clean points in the current mini-batch.
        batch_size = x0.shape[0]
        # Draw one independent integer timestep in [0, n_steps) for every point.
        ## The resulting tensor has shape (B,) and is created on x0's device.
        ### Different points can train on different noise levels within one batch.
        t = torch.randint(0, self.schedule.n_steps, (batch_size,), device=x0.device)
        # Draw fresh target noise with shape (B, D) each time the loss is evaluated.
        noise = torch.randn_like(x0)

        # Construct the noisy input using exactly the noise the model must predict.
        x_t = self.q_sample(x0, t, noise)
        # Predict that noise from the noisy points and their timestep indices.
        ## Both predicted_noise and noise have shape (B, D).
        predicted_noise = model(x_t, t)

        # Square each prediction error and average over both points and coordinates.
        ## The scalar loss retains its computation graph for loss.backward().
        return (noise - predicted_noise).square().mean()

    # =============================================================================
    # Reverse process  p(x_{t-1} | x_t)
    # =============================================================================

    # Sampling uses the current parameters without building a gradient graph.
    ## This reduces memory use while drawing points from the trained model.
    @torch.no_grad()
    def p_sample(
        self,
        model: ConditionalModel,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Single reverse step: sample x_{t-1} from p(x_{t-1} | x_t).

        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
                  + sigma_t * z

        Args:
            model: Trained noise-prediction network.
            x_t: Current noisy samples, shape (B, D).
            t: Current timestep (scalar int).

        Returns:
            Denoised samples x_{t-1}, shape (B, D).
        """
        # Repeat the single reverse timestep for all B points, giving shape (B,).
        ## Integer indices select the model's learned timestep embeddings.
        ### Keep those indices on the same device as the current points.
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)

        # Read the scalar schedule values for this transition.
        ## Scalars broadcast across all points and their D coordinates below.
        alpha_t = self.schedule.alphas[t]
        beta_t = self.schedule.betas[t]
        alpha_bar_t = self.schedule.alphas_bar[t]

        # Estimate the noise component in the current points, with shape (B, D).
        eps_theta = model(x_t, t_tensor)

        # Scale the predicted noise by the coefficient in the DDPM reverse mean.
        eps_coeff = beta_t / torch.sqrt(1 - alpha_bar_t)
        # Subtract that contribution and undo the transition's signal scaling.
        ## mean is the predicted Gaussian reverse transition's mean, shape (B, D).
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - eps_coeff * eps_theta)

        # Keep intermediate reverse transitions stochastic, then finish at the mean.
        if t > 0:
            # Choose the fixed reverse variance beta_t, so its standard deviation is sqrt(beta_t).
            ## This implementation uses beta_t rather than the forward posterior variance.
            sigma_t = torch.sqrt(beta_t)
            # Draw an independent standard Gaussian perturbation for this update.
            z = torch.randn_like(x_t)
            # Scale the perturbation by its standard deviation and add it to the mean.
            return mean + sigma_t * z
        # At index 0, return the final denoised coordinates without an extra noise draw.
        return mean

    # Disable gradient recording throughout the complete reverse chain as well.
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: ConditionalModel,
        shape: tuple,
        device: torch.device | str = "cpu",
        return_trajectory: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Full reverse sampling chain: x_T -> x_{T-1} -> ... -> x_0.

        Args:
            model: Trained noise-prediction network.
            shape: Shape of samples to generate, e.g. (1000, 2).
            device: Device for computation.
            return_trajectory: If True, return list of all intermediate x_t.

        Returns:
            Generated samples x_0 (or full trajectory list).
        """
        # Start with independent standard Gaussian coordinates, for example (1000, 2).
        ## The caller should place the model and schedule on this device beforehand.
        ### The short default schedule retains substantial signal at its terminal state;
        ### this Gaussian initialization is an approximation there (see NoiseSchedule).
        x = torch.randn(shape, device=device)
        # Optionally retain the initial state on CPU for later plotting.
        ## .cpu() transfers GPU tensors to CPU and returns CPU tensors as-is.
        ### No explicit clone is made here; each reverse update below creates a new tensor.
        trajectory = [x.cpu()] if return_trajectory else None

        # Visit timestep indices in descending order: n_steps - 1, ..., 0.
        for t in reversed(range(self.schedule.n_steps)):
            # Replace the current points with the result of one reverse transition.
            x = self.p_sample(model, x, t)
            # Store each updated state only when the caller requests the full history.
            if return_trajectory:
                trajectory.append(x.cpu())

        # A complete history contains n_steps + 1 states, including the initial noise.
        ## trajectory[0] is the start; trajectory[-1] contains the generated points.
        if return_trajectory:
            return trajectory
        # Otherwise, return only the final tensor on the requested computation device.
        return x
