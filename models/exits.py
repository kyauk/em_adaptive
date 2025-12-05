import torch.nn as nn


class ExitClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExitClassifier, self).__init__()
        
        # MLP Classifier (Linear -> ReLU -> Linear)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        # If input is 4D (B, C, H, W), apply GAP using torch.mean to match caching
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])  # Global Average Pooling
        
        return self.classifier(x)