import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, input_dim):
        """
        Lightweight linear router network.
        Args:
            input_dim: Dimension of input features (e.g., 64, 128, 256)

        Model Architecture:
        Input (B, C, H, W) -> Global Average Pooling -> (B, C) -> Linear -> (B, 1) (Logits)
        """
        super(Router, self).__init__()
        self.net = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Returns logits of exiting (unbounded).
        """
        # Normalize input shape (B, C, H, W) -> Global Average Pooling -> (B, C)
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        # Pass through network
        return self.net(x)
