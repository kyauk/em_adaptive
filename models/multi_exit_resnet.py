"""

Multi-Exit ResNet-18 Architecture

Here we are implementing a ResNet-18 architecture with multiple exit points. Resnet backbone is frozen
and exit classifiers are added after each residual block group.

"""
from models.exits import ExitClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiExitResNet(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super(MultiExitResNet, self).__init__()

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # Load standard ResNet-18 structure (weights don't matter as we'll modify/train)
        resnet18 = models.resnet18(weights=None)

        # Modify for CIFAR-10 (32x32 images)
        # Replace 7x7 conv stride 2 with 3x3 conv stride 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        # Remove maxpool for CIFAR-10 to preserve spatial dimensions
        self.maxpool = nn.Identity()

        # Residual block groups
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        # Freeze the backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

        # This is where we'll add exit classfiers
        self.exit1 = ExitClassifier(64, num_classes)
        self.exit2 = ExitClassifier(128, num_classes)
        self.exit3 = ExitClassifier(256, num_classes)
        self.exit4 = ExitClassifier(512, num_classes)

    def _freeze_backbone(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        
        print("This message is to alert the user that the backbone has been frozen.")

    # build the forward pass
    def forward(self, x, return_all_exits=False):
        # initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # forward through residual blocks
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        if return_all_exits:
            return {
                'exit1': self.exit1(f1),
                'exit2': self.exit2(f2),
                'exit3': self.exit3(f3),
                'exit4': self.exit4(f4),
            }
        else:
            return self.exit4(f4)

    # build forward pass up to a specific layer, and extract that specific layer's features
    def forward_to_layer(self, x, layer_num):
        # initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # forward through residual blocks
        if layer_num >=1:
            x = self.layer1(x)
        if layer_num >=2:
            x = self.layer2(x)
        if layer_num >=3:
            x = self.layer3(x)
        if layer_num >=4:
            x = self.layer4(x)
        
        return x

    def extract_all_features(self, x):
        # avoid extra computation and bookkeeping
        with torch.no_grad():
            # initial convolution
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            # extract features at each layer
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            
            return {
                'layer1': f1.cpu(),
                'layer2': f2.cpu(),
                'layer3': f3.cpu(),
                'layer4': f4.cpu()
            }


