"""
EM Routing Algorithm

Implements the Expectation-Maximization (EM) algorithm to find optimal routing assignments.
Since our backbone and exit classifiers are frozen, we only need the E-step to compute
the "ground truth" assignments for the router to learn.

Utility = I(correct) - lambda * Cost
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

class EMRouting:
    def __init__(self, model, lambda_val=0.5):
        """
        Args:
            model: The MultiExitResNet model (frozen)
            lambda_val: Trade-off parameter. Higher lambda = more penalty for late exits.
        """
        self.model = model
        self.lambda_val = lambda_val
        self.device = next(model.parameters()).device
        
        # TODO: Define computational costs for each exit (normalized 0 to 1)
        # Suggestion: [0.25, 0.50, 0.75, 1.00]
        self.costs = None 

    def get_costs(self):
        return self.costs

    def e_step(self, features_tuple, targets):
        """
        Compute soft assignment probabilities P(z=k|x) for a batch.
        
        Args:
            features_tuple: (f1, f2, f3, f4) tuple of feature tensors
            targets: Ground truth labels [B]
            
        Returns:
            assignments: Soft assignments [B, 4]
        """
        # TODO: Implement the E-step logic
        # 1. Get logits from all exits
        # 2. Calculate P(correct | x, exit_k)
        # 3. Calculate Utility = P(correct) - lambda * Cost
        # 4. Compute Softmax(Utility) -> Assignments
        
        pass

    def run(self, dataloader):
        """
        Run E-step over the entire dataset to generate assignments.
        """
        self.model.eval()
        all_assignments = []
        all_targets = []
        
        print(f"Generating assignments (lambda={self.lambda_val})...")
        
        with torch.no_grad():
            for f1, f2, f3, f4, labels in tqdm(dataloader):
                # Move to device
                f1, f2, f3, f4 = f1.to(self.device), f2.to(self.device), f3.to(self.device), f4.to(self.device)
                labels = labels.to(self.device)
                
                features = (f1, f2, f3, f4)
                
                # Compute assignments
                batch_assignments = self.e_step(features, labels)
                
                all_assignments.append(batch_assignments.cpu())
                all_targets.append(labels.cpu())
                
        return torch.cat(all_assignments, dim=0), torch.cat(all_targets, dim=0)
