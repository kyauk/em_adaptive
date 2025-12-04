import torch.nn as nn


class ExitClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExitClassifier, self).__init__()
        # Use nn.Linear only, pooling will be done in forward()
        # We use torch.mean() to match feature_cache.py and Router
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # If input is 4D (B, C, H, W), apply GAP using torch.mean to match caching
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])  # Global Average Pooling
        # If input is 2D (B, C), it's already pooled (e.g., from cache)
        return self.linear(x)