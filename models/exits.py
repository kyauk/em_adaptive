import torch.nn as nn


class ExitClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExitClassifier, self).__init__()
        # Split into pooling/flatten and linear for flexibility
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # If input is 4D (B, C, H, W), apply pooling and flatten
        if x.dim() == 4:
            x = self.pool(x)
            x = self.flatten(x)
        # If input is 2D (B, C), it's already pooled (e.g., from cache)
        return self.linear(x)