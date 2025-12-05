"""
EM Routing Algorithm

Implements the Expectation-Maximization (EM) algorithm to find optimal routing assignments.
Since our backbone and exit classifiers are frozen, we only need the E-step to compute
the "ground truth" assignments for the router to learn.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

eps = 1e-10
class EMRouting:
    def __init__(self, model, lambda_val=0.05):
        """
        Args:
            model: The MultiExitResNet model (frozen)
            lambda_val: Trade-off parameter. Higher lambda = more penalty for late exits.
        """
        self.model = model
        self.lambda_val = lambda_val
        self.device = next(model.parameters()).device
        
        self.costs = torch.tensor([0.25, 0.50, 0.75, 1.00]).to(self.device)
        self.priors = (torch.ones(4) / 4).to(self.device)
    def get_costs(self):
        return self.costs

    def e_step(self, features_tuple, labels):
        """
        Compute soft assignment probabilities P(z=k|x) for a batch.
        
        Args:
            features_tuple: (f1, f2, f3, f4) tuple of feature tensors
            labels: Ground truth labels [B]
            
        Returns:
            assignments: Soft assignments [B, 4]
        """
        # Get logits from all exits
        # each feature tensor is [B, C, H, W]
        # each out is [B, ]
        f1, f2, f3, f4 = features_tuple
        out1 = self.model.exit1(f1)
        out2 = self.model.exit2(f2)
        out3 = self.model.exit3(f3)
        out4 = self.model.exit4(f4)
        logits_tuple = torch.stack([out1, out2, out3, out4], dim=1)
        # Calculate P(correct | x, exit_k)
        # expand labels to match the shape of logits_tuple
        labels = labels.unsqueeze(1).unsqueeze(2)
        labels = labels.expand(-1, 4, -1)   
        # Use log-probabilities for numerical stability
        # p_correct = log(P(correct | x, exit_k)) + log(P(exit_k))
        log_probs = F.log_softmax(logits_tuple, dim=2)
        p_correct_given_exit = log_probs.gather(2, labels).squeeze(2)
        p_correct = p_correct_given_exit + torch.log(self.priors)[None, :]
        
        # Calculate numerator for posterior: log(P(z=k, y|x)) = log(P(y|x,z)) + log(P(z)) - lambda * Cost
        numerator = p_correct - (self.lambda_val * self.costs[None, :])
        # Adding Denominator (which happens to create a softmax)
        assignments = F.softmax(numerator, dim=1)
        return assignments
        
    def m_step(self, all_assignments):
        """
        Update priors pi_k based on aggregated assignments.
        
        Args:
            all_assignments: [N, 4] tensor of soft assignments
        """
        # Calculate mean of assignments across the dataset 
        new_priors = all_assignments.mean(dim=0)
        new_priors += eps
        new_priors = new_priors / new_priors.sum()
        # Update self.priors with these new values
        self.priors = new_priors.to(self.device)
        # Print the new priors to track progress
        print(f"Updated priors: {self.priors.cpu().numpy()}")
        return

    def run(self, dataloader, iterations=5):
        """
        Run EM algorithm over the dataset.
        """
        self.model.eval()
        
        print(f"Running EM (lambda={self.lambda_val}) for {iterations} iterations...")
        
        for i in range(iterations):
            all_assignments = []
            all_labels = []
            # E-Step: Pass over entire dataset
            with torch.no_grad():
                for f1, f2, f3, f4, labels in tqdm(dataloader):
                    # Move to device
                    f1, f2, f3, f4 = f1.to(self.device), f2.to(self.device), f3.to(self.device), f4.to(self.device)
                    labels = labels.to(self.device)
                    
                    features = (f1, f2, f3, f4)
                    
                    # Compute assignments
                    batch_assignments = self.e_step(features, labels)
                    
                    all_assignments.append(batch_assignments.cpu())
                    all_labels.append(labels.cpu())
            
            # Concatenate all
            full_assignments = torch.cat(all_assignments, dim=0)
            
            # M-Step: Update priors
            self.m_step(full_assignments) 
            
            # Optional: Check convergence or print stats
            exit_counts = full_assignments.sum(dim=0)
            print(f"Iter {i+1}: Exit distribution: {exit_counts.cpu().numpy().astype(int)}")
                
        return full_assignments, torch.cat(all_labels, dim=0)
