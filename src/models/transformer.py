import torch

import torch.nn as nn

class Transformer(nn.Module):
    """
    Transformer encoder model with projection and classification head.

    Args:
            input_dim (int): Dimension of input features.
            d_model (int): Dimension of the model (projection size).
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            activation (str): Activation function for encoder layers.
            num_layers (int): Number of encoder layers.
            num_classes (int): Number of output classes.
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, activation: str, num_layers: int, num_classes: int) -> None:
        super().__init__()
        
        self.projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).
            padding_mask (torch.Tensor, optional): Bool mask of shape (batch, seq_len),
                True for padding positions. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, num_classes).
        """
        x = self.projection(x)
        
        if padding_mask is not None:
            src_key_padding_mask = padding_mask.bool()
        else:
            src_key_padding_mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        y = self.classifier(x)
        
        return y