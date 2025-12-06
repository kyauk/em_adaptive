
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class RouterMetrics:
    def __init__(self):
        self.router_probs_list = []
        self.em_probs_list = []
        self.history = {
            'accuracy': [],
            'ce_loss': [],
            'epochs': []
        }

    def reset(self):
        self.router_probs_list = []
        self.em_probs_list = []

    def update(self, router_logits, em_targets):
        """
        Args:
            router_logits: [B, 1] tensor of logit outputs from router
            em_targets: [B, 1] tensor of target probabilities (cond prob)
        """
        # Convert logits to probs
        probs = torch.sigmoid(router_logits)
        self.router_probs_list.append(probs.detach().cpu())
        self.em_probs_list.append(em_targets.detach().cpu())

    def compute(self):
        """
        Computes metrics over the accumulated batches.
        Returns dict: {'accuracy': float, 'ce_loss': float}
        """
        if not self.router_probs_list:
            return {'accuracy': 0.0, 'ce_loss': 0.0}

        # Concatenate all batches
        # [N, 1]
        all_router_probs = torch.cat(self.router_probs_list, dim=0)
        all_em_probs = torch.cat(self.em_probs_list, dim=0)
        
        # 1. Accuracy (Matching Hard Decisions)
        # Using 0.5 threshold for both as the "Hard" decision boundary
        router_hard = (all_router_probs > 0.5).float()
        em_hard = (all_em_probs > 0.5).float()
        
        accuracy = (router_hard == em_hard).float().mean().item()
        
        # 2. Cross Entropy
        # BCE = -(y * log(p) + (1-y) * log(1-p))
        # Add epsilon for numerical stability inside log
        eps = 1e-8
        p = all_router_probs
        y = all_em_probs
        ce_loss = -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean().item()
        
        return {'accuracy': accuracy, 'ce_loss': ce_loss}

    def record_epoch(self, epoch):
        metrics = self.compute()
        self.history['accuracy'].append(metrics['accuracy'])
        self.history['ce_loss'].append(metrics['ce_loss'])
        self.history['epochs'].append(epoch)
        self.reset() # Reset for next epoch
        return metrics

    def plot(self, save_dir="results"):
        """
        Plots accuracy and CE metrics over epochs.
        """
        os.makedirs(save_dir, exist_ok=True)
        epochs = self.history['epochs']
        if not epochs:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (Hard Match)', color=color)
        ax1.plot(epochs, self.history['accuracy'], color=color, marker='o', label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.05)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Cross Entropy Loss', color=color)
        ax2.plot(epochs, self.history['ce_loss'], color=color, marker='x', linestyle='--', label='CE Loss')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Router Approximation Quality vs EM Targets')
        fig.tight_layout()
        
        output_path = os.path.join(save_dir, 'router_training_metrics.png')
        plt.savefig(output_path)
        plt.close()
        # print(f"Metrics plot saved using matplotlib Agg backend to {output_path}")
