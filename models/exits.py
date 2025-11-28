import torch.nn as nn


class ExitClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ExitClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)