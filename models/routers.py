import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, use_mlp=True):
        """
        Lightweight 2-layer router network.
        Args:
            input_dim: Dimension of input features (e.g., 64, 128, 256)
            hidden_dim: Hidden layer dimension
            use_mlp: Whether to use a 2-layer MLP (True) or a simple linear layer (False)
        """
        super(Router, self).__init__()
        if use_mlp:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()
            )
        


    def forward(self, x):
        """
        Returns probability of exiting (0 to 1).
        """
        # TODO: Implement forward pass
        # 1. Handle input shape (B, C, H, W) -> Global Average Pooling -> (B, C)
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        # 2. Pass through network
        return self.net(x)
