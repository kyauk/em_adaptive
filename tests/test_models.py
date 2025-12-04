import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.routers import Router
from models.multi_exit_resnet import MultiExitResNet

class TestModels(unittest.TestCase):
    def test_router_shapes(self):
        # Testing Router forward pass with different input shapes
        batch_size = 4
        input_dim = 64
        
        # Test MLP Router
        router = Router(input_dim=input_dim, use_mlp=True)
        # Need to check if GAP is working correctly
        # Case 1: 4D Input (B, C, H, W)
        x_4d = torch.randn(batch_size, input_dim, 32, 32)
        out = router(x_4d)
        self.assertEqual(out.shape, (batch_size, 1))
        self.assertTrue((out >= 0).all() and (out <= 1).all()) # Sigmoid output
        
        # Case 2: 2D Input (B, C)
        x_2d = torch.randn(batch_size, input_dim)
        out = router(x_2d)
        self.assertEqual(out.shape, (batch_size, 1))

    def test_multiexit_resnet_shapes(self):
        # Testing MultiExitResNet forward pass and feature extraction
        model = MultiExitResNet(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        
        # Test full forward
        out = model(x)
        self.assertEqual(out.shape, (2, 10))
        
        # Test all exits
        outputs = model(x, return_all_exits=True)
        self.assertEqual(len(outputs), 4)
        for k, v in outputs.items():
            self.assertEqual(v.shape, (2, 10))
            
        # Test feature extraction
        features = model.extract_all_features(x)
        self.assertEqual(len(features), 4)
        self.assertEqual(features['layer1'].shape[1], 64)
        self.assertEqual(features['layer4'].shape[1], 512)

if __name__ == '__main__':
    unittest.main()
