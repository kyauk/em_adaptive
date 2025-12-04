import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.routers import Router
from algorithms.em_routing import EMRouting
from models.multi_exit_resnet import MultiExitResNet

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = MultiExitResNet(num_classes=10, freeze_backbone=True)
        self.model.to(self.device)
        
        # Create dummy cached features
        batch_size = 10
        self.f1 = torch.randn(batch_size, 64, 32, 32)
        self.f2 = torch.randn(batch_size, 128, 16, 16)
        self.f3 = torch.randn(batch_size, 256, 8, 8)
        self.f4 = torch.randn(batch_size, 512, 4, 4)
        self.labels = torch.randint(0, 10, (batch_size,))
        
        self.dataset = TensorDataset(self.f1, self.f2, self.f3, self.f4, self.labels)
        self.loader = DataLoader(self.dataset, batch_size=5)

    def test_full_training_loop(self):
        # Simulating entire router training pipeline
        print("\nTesting Full Training Loop...")
        
        # 1. EM Step
        em = EMRouting(self.model)
        assignments, _ = em.run(self.loader, iterations=2)
        hard_assignments = torch.argmax(assignments, dim=1)
        # Expect size to be batch size
        self.assertEqual(hard_assignments.shape, (batch_size,))
        
        # 2. Router Training Step
        router1 = Router(input_dim=64, use_mlp=True).to(self.device)
        optimizer = optim.Adam(router1.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Create router dataset
        router_dataset = TensorDataset(self.f1, hard_assignments)
        router_loader = DataLoader(router_dataset, batch_size=5)
        
        # Train for 1 epoch
        router1.train()
        for batch in router_loader:
            bf1, b_targets = batch
            # Target: 1 if Exit 0 is optimal
            target = (b_targets == 0).float().unsqueeze(1)
            
            optimizer.zero_grad()
            pred = router1(bf1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
        print("Training loop ran successfully")
        
        # 3. Save Checkpoint (Mock)
        os.makedirs('tests/checkpoints_test', exist_ok=True)
        torch.save(router1.state_dict(), 'tests/checkpoints_test/router1.pth')
        self.assertTrue(os.path.exists('tests/checkpoints_test/router1.pth'))
        
    def tearDown(self):
        # Clean up
        if os.path.exists('tests/checkpoints_test'):
            shutil.rmtree('tests/checkpoints_test')

if __name__ == '__main__':
    unittest.main()
