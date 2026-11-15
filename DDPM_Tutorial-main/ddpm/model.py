"""
Neural network architecture for DDPM noise prediction.

Uses a simple MLP with time-step conditioning via learned embeddings.
For 2D data (like Swiss Roll), a lightweight architecture suffices —
no U-Net required.
"""

# =============================================================================
# Imports
# =============================================================================

# Load PyTorch's tensor type for the input and output annotations below.
import torch
# Load trainable neural-network components and the common Module base class.
import torch.nn as nn
# Load activation functions, including Softplus.
import torch.nn.functional as F


# =============================================================================
# Conditional Linear Layer: Learn a Separate Feature Scale for Each Timestep
# =============================================================================

# Inherit from nn.Module so PyTorch can collect the layer's trainable parameters.
class ConditionalLinear(nn.Module):
    """
    Linear layer conditioned on a discrete timestep.

    Instead of directly passing the timestep as an input feature,
    each timestep learns a multiplicative scaling vector (gamma) via
    an embedding table.  This allows the same layer weights to behave
    differently at every diffusion step.

    forward(x, t) -> gamma_t * (Wx + b)
    """

    # Set the input width, output width, and number of valid timestep indices.
    def __init__(self, in_features: int, out_features: int, num_steps: int):
        # Initialize Module's parameter and sublayer bookkeeping before assigning layers.
        super().__init__()

        # Learn an affine transformation: out = x W^T + b.
        ## Input shape (B, in_features) becomes (B, out_features), where B is batch size.
        self.linear = nn.Linear(in_features, out_features)

        # Create one trainable scale vector for every timestep index.
        ## The embedding table has shape (num_steps, out_features).
        ### A point at timestep t uses row t of this table to scale its hidden features.
        self.embed = nn.Embedding(num_steps, out_features)

        # Fill the embedding weights in place with uniform random values in [0, 1).
        ## This initialization happens before training; the optimizer later learns the scales.
        self.embed.weight.data.uniform_()

    # Calling layer(x, t) invokes this computation for a batch of points.
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Apply the same affine transformation to every point in the batch.
        out = self.linear(x)

        # Look up the scale vector for each point's timestep.
        ## Integer t has shape (B,); gamma has shape (B, out_features).
        gamma = self.embed(t)

        # Match the scale tensor to the output shape and multiply element by element.
        ## out.size(-1) is the feature width; -1 lets view infer the batch dimension.
        ### The timestep changes the feature scales while the linear weights are shared.
        return gamma.view(-1, out.size(-1)) * out


# =============================================================================
# Conditional MLP: Predict the Gaussian Noise Added to Each Point
# =============================================================================

class ConditionalModel(nn.Module):
    """
    Simple MLP that predicts the noise epsilon_theta(x_t, t).

    Architecture: 3 conditional hidden layers and 1 ordinary linear output
    layer (2 -> 128 -> 128 -> 128 -> 2), with softplus hidden activations.
    The output dimension equals the data dimension (2 for Swiss Roll).
    """

    # Use one embedding entry per schedule index and 128 hidden features by default.
    def __init__(self, n_steps: int, data_dim: int = 2, hidden_dim: int = 128):
        # Initialize the parent module before registering this model's child layers.
        super().__init__()

        # Expand each noisy point from data_dim coordinates into hidden_dim learned features.
        ## With the defaults, the tensor shape changes from (B, 2) to (B, 128).
        self.lin1 = ConditionalLinear(data_dim, hidden_dim, n_steps)

        # Refine those hidden features using two more timestep-conditioned transformations.
        ## Both transformations keep the shape (B, hidden_dim).
        self.lin2 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin3 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)

        # Map the hidden features back to data_dim predicted noise coordinates.
        ## The final layer is an ordinary Linear layer; conditioning occurs in the hidden layers.
        self.lin4 = nn.Linear(hidden_dim, data_dim)

    # Predict noise from noisy coordinates x and one integer timestep per point.
    ## x has shape (B, data_dim); t has shape (B,).
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Apply the first conditioned layer, then introduce a smooth nonlinearity.
        ## Softplus(u) = log(1 + exp(u)); the result has shape (B, hidden_dim).
        x = F.softplus(self.lin1(x, t))

        # Repeat the conditioned transformation and activation through the hidden stack.
        x = F.softplus(self.lin2(x, t))
        x = F.softplus(self.lin3(x, t))

        # Return predicted noise with shape (B, data_dim), matching the added noise tensor.
        ## Leave this output unconstrained so its coordinates can be positive or negative.
        return self.lin4(x)
