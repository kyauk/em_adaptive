import unittest
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.em_routing import EMRouting
from models.multi_exit_resnet import MultiExitResNet

class TestEMRouting(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MultiExitResNet(num_classes=10, freeze_backbone=True)
        self.model.to(self.device)
        self.em = EMRouting(self.model, lambda_val=0.5)

    def test_initialization(self):
        """Test that EM initializes with correct priors and costs."""
        self.assertTrue(torch.allclose(self.em.priors, torch.tensor([0.25, 0.25, 0.25, 0.25])))
        self.assertTrue(torch.allclose(self.em.costs, torch.tensor([0.25, 0.50, 0.75, 1.00])))

    def test_e_step_shapes(self):
        """Test E-step output shapes and probability properties."""
        batch_size = 4
        # Create dummy features
        f1 = torch.randn(batch_size, 64, 32, 32)
        f2 = torch.randn(batch_size, 128, 16, 16)
        f3 = torch.randn(batch_size, 256, 8, 8)
        f4 = torch.randn(batch_size, 512, 4, 4)
        features = (f1, f2, f3, f4)
        labels = torch.randint(0, 10, (batch_size,))

        assignments = self.em.e_step(features, labels)
        
        # Check shape
        self.assertEqual(assignments.shape, (batch_size, 4))
        
        # Check sum to 1 (valid probabilities)
        sums = assignments.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones(batch_size), atol=1e-5))

    def test_m_step_update(self):
        """Test that M-step correctly updates priors."""
        # Create dummy assignments (Batch=2, Exits=4)
        # Sample 1: 100% Exit 0
        # Sample 2: 100% Exit 1
        assignments = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        self.em.m_step(assignments)
        
        # New priors should be mean: [0.5, 0.5, 0.0, 0.0]
        expected_priors = torch.tensor([0.5, 0.5, 0.0, 0.0])
        self.assertTrue(torch.allclose(self.em.priors.cpu(), expected_priors))

if __name__ == '__main__':
    unittest.main()
