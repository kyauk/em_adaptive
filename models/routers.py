import torch
import torch.nn as nn

class Router(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Lightweight router network.
        Args:
            input_dim: Dimension of input features (e.g., 64, 128, 256)
            hidden_dim: Hidden layer dimension
        """
        super(Router, self).__init__()
        
        # TODO: Define the router architecture
        # It should take input_dim -> hidden_dim -> 1 (scalar probability)
        # Use ReLU and Sigmoid activation
        self.net = None 
        
    def forward(self, x):
        """
        Returns probability of exiting (0 to 1).
        """
        # TODO: Implement forward pass
        # 1. Handle input shape (B, C, H, W) -> Global Average Pooling -> (B, C)
        # 2. Pass through network
        pass
