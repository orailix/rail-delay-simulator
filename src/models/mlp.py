import torch

import torch.nn as nn

class MLP(nn.Module):
    """
    Simple feedforward multi-layer perceptron (MLP).
    
    Args:
            input_dim (int): Input feature dimension.
            hidden_dims (list): List of hidden layer dimensions.
            output_dim (int): Output dimension.
            activation (str): Name of the activation function from torch.nn (e.g., "ReLU").
            dropout (float): Dropout probability (0.0–1.0).
    """
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, activation: str, dropout: float) -> None:
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        act_cls = activations.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation: {activation}")

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.net(x)
