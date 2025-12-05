import torch
import torch.nn.functional as F
import random
from models.routers import Router
"""
Run Evaluation experiments, including:
1. Standard ResNet (Lower Bound)
2. Multi-Exit ResNet (Fixed, Random)
3. BranchyNet (Entropy-based)
4. Oracle (Upper Bound)
5. EM Routing (Proposed Method)
"""

class Evaluator:
    def __init__(self, model):
        self.model = model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # Initialize cost tensor onto device
        self.cost = torch.tensor([0.25, 0.50, 0.75, 1.00]).to(self.device)
    
    def eval_resnet(self, dataloader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_cost = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images, return_all_exits=False)
                total_samples += labels.size(0)
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                # Cost is 1.0 for each image as it is going til the 4th exit
                total_cost += 1.0*images.size(0)
        
        acc = total_correct/total_samples
        avg_cost = total_cost/total_samples
        if isinstance(avg_cost, torch.Tensor):
            avg_cost = avg_cost.item()
        print(f"Standard ResNet-18: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": float(acc), "cost": float(avg_cost)}

    def eval_multiexit_resnet_fixed(self, dataloader):
        self.model.eval()
        exit_stats = {f"exit{i+1}": {"correct": 0, "total": 0, "cost": 0} for i in range(4)}
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images, return_all_exits=True)
                
                for i, (exit_name, output) in enumerate(outputs.items()):
                    exit_stats[exit_name]["total"] += labels.size(0)
                    exit_stats[exit_name]["correct"] += (output.argmax(dim=1) == labels).sum().item()
                    exit_stats[exit_name]["cost"] += self.cost[i].item() * images.size(0)
                
        results = {}
        for exit_name, stats in exit_stats.items():
            acc = stats['correct']/stats['total']
            avg_cost = stats['cost']/stats['total']
            print(f"Fixed on {exit_name}: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
            results[exit_name] = {"accuracy": float(acc), "cost": float(avg_cost)}
        return results

    def eval_multiexit_resnet_random(self, dataloader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_cost = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images, return_all_exits=True)
                
                # Randomly select an exit for each image in the batch
                exit_idx = random.randint(0, 3)
                exit_name = f"exit{exit_idx+1}"
                
                total_samples += labels.size(0)
                total_correct += (outputs[exit_name].argmax(dim=1) == labels).sum().item()
                total_cost += self.cost[exit_idx].item() * images.size(0)
                
        acc = total_correct/total_samples
        avg_cost = total_cost/total_samples
        if isinstance(avg_cost, torch.Tensor):
            avg_cost = avg_cost.item()
        print(f"Random Routing: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": float(acc), "cost": float(avg_cost)}

    def eval_branchynet(self, dataloader, threshold=0.5):
        """
        BranchyNet strategy: Exit if entropy < threshold.
        Entropy is defined as e(p) - sum of p * log(p) for each exit
        Otherwise continue to next exit.
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_cost = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images, return_all_exits=True)
                
                # Calculate entropy for each exit (lower is more confident)
                # entropy = -sum(p * log(p))
                entropies = []
                for i in range(4):
                    probs = F.softmax(outputs[f'exit{i+1}'], dim=1)
                    log_probs = F.log_softmax(outputs[f'exit{i+1}'], dim=1)
                    entropy = -(probs * log_probs).sum(dim=1)
                    entropies.append(entropy)

                batch_size = images.size(0)
                final_preds = torch.zeros(batch_size, dtype=torch.long).to(self.device)
                
                # Mask for Exit 1: entropy < threshold
                mask1 = (entropies[0] < threshold)
                total_cost += self.cost[0] * mask1.sum().item()
                
                # Mask for Exit 2: entropy < threshold AND didn't exit at Exit 1
                mask2 = (entropies[1] < threshold) * (mask1 == 0)
                total_cost += self.cost[1] * mask2.sum().item()
                
                # Mask for Exit 3: entropy < threshold AND didn't exit at Exit 1, 2
                mask3 = (entropies[2] < threshold) * (mask1 == 0) * (mask2 == 0)
                total_cost += self.cost[2] * mask3.sum().item()
                
                # Exit 4: Everyone else
                mask4 = (mask1 == 0) * (mask2 == 0) * (mask3 == 0)
                total_cost += self.cost[3] * mask4.sum().item()
                
                # Fill predictions
                final_preds[mask1] = outputs['exit1'][mask1].argmax(dim=1)
                final_preds[mask2] = outputs['exit2'][mask2].argmax(dim=1)
                final_preds[mask3] = outputs['exit3'][mask3].argmax(dim=1)
                final_preds[mask4] = outputs['exit4'][mask4].argmax(dim=1)

                total_samples += batch_size
                total_correct += (final_preds == labels).sum().item()
        
        acc = total_correct/total_samples
        avg_cost = total_cost/total_samples
        if isinstance(avg_cost, torch.Tensor):
            avg_cost = avg_cost.item()
        print(f"BranchyNet: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": float(acc), "cost": float(avg_cost)}

    def eval_oracle(self, dataloader):
        """
        Cheater's strategy: Oracle ball and select the earliest exit that is correct.
        If none are correct, select the last exit.
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_cost = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images, return_all_exits=True)
                
                # batch_size = images.size(0)
                # sample_costs = torch.zeros(batch_size).to(self.device)
                # Default to incorrect and max cost
                # sample_correct = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
                # sample_costs[:] = self.cost[3] 
                
                # Check exits in order
                # This logic was slightly flawed in original code (accumulating cost weirdly)
                # Simplified Oracle:
                # For each sample, find first correct exit. If none, exit 4.
                
                batch_size = images.size(0)
                
                # Get correctness for each exit [B, 4]
                correct_mask = torch.stack([
                    outputs['exit1'].argmax(1) == labels,
                    outputs['exit2'].argmax(1) == labels,
                    outputs['exit3'].argmax(1) == labels,
                    outputs['exit4'].argmax(1) == labels
                ], dim=1) # [B, 4]
                
                # Find first True index
                # We can use (correct_mask.cumsum(1) > 0) to find where we first hit a correct one
                # But simpler:
                
                # Default cost = exit 4 cost
                batch_cost = torch.full((batch_size,), self.cost[3].item(), device=self.device)
                batch_correct = correct_mask[:, 3] # Default to exit 4 correctness
                
                # Check 3, then 2, then 1 (reverse order to overwrite)
                # Actually forward order is better if we break? Vectorized break is hard.
                # Let's use masks.
                
                # Exit 1 correct?
                mask1 = correct_mask[:, 0]
                batch_cost[mask1] = self.cost[0]
                batch_correct[mask1] = True
                
                # Exit 2 correct AND Exit 1 NOT correct?
                mask2 = correct_mask[:, 1] & (~mask1)
                batch_cost[mask2] = self.cost[1]
                batch_correct[mask2] = True
                
                # Exit 3 correct AND Exit 1,2 NOT correct?
                mask3 = correct_mask[:, 2] & (~mask1) & (~mask2)
                batch_cost[mask3] = self.cost[2]
                batch_correct[mask3] = True
                
                # Else (Exit 4) -> already set defaults.
                # But wait, if Exit 4 is correct, batch_correct is True.
                # If Exit 4 is NOT correct, and 1-3 NOT correct, batch_correct is False.
                # My default `batch_correct = correct_mask[:, 3]` handles this.
                
                total_cost += batch_cost.sum().item()
                total_correct += batch_correct.sum().item()
                total_samples += batch_size

        acc = total_correct/total_samples
        avg_cost = total_cost/total_samples
        if isinstance(avg_cost, torch.Tensor):
            avg_cost = avg_cost.item()
        print(f"Oracle: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": float(acc), "cost": float(avg_cost)}

    def eval_em_routing(self, dataloader, routers, threshold=0.5):
        """
        EM Routing strategy: Use trained routers to decide when to exit.
        routers: list of 4 router models (or dict)
        threshold: Probability threshold for exiting (default: 0.5)
        """
        self.model.eval()
        for r in routers:
            r.eval()
            
        total_correct = 0
        total_samples = 0
        total_cost = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)
                
                # outputs are dict: {'exit1': [B, 10], ...}
                # features are dict: {'feature1': [B, C, H, W], ...}
                outputs, features = self.model(images, return_all_exits=True, return_features=True)
                
                # Initialize masks (False = hasn't exited yet)
                has_exited = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
                final_preds = torch.zeros(batch_size, dtype=torch.long).to(self.device)
                
                # Exit 1
                f1 = features['feature1']
                # Router returns logits, apply sigmoid to get probability
                p1 = torch.sigmoid(routers[0](f1)).squeeze(1) # [B, 1] -> [B]
                # Exit if p1 > threshold AND hasn't exited
                mask1 = (p1 > threshold) & (~has_exited)
                # apply masking
                if mask1.any():
                    preds1 = outputs['exit1'][mask1].argmax(dim=1)
                    final_preds[mask1] = preds1
                    total_cost += self.cost[0].item() * mask1.sum().item()
                    # marks all the samples that exited at Exit 1
                    has_exited |= mask1

                # Exit 2
                f2 = features['feature2']
                p2 = torch.sigmoid(routers[1](f2)).squeeze(1)
                mask2 = (p2 > threshold) & (~has_exited)
                # apply masking
                if mask2.any():
                    preds2 = outputs['exit2'][mask2].argmax(dim=1)
                    final_preds[mask2] = preds2
                    total_cost += self.cost[1].item() * mask2.sum().item()
                    has_exited |= mask2

                # Exit 3
                f3 = features['feature3']
                p3 = torch.sigmoid(routers[2](f3)).squeeze(1)
                mask3 = (p3 > threshold) & (~has_exited)
                # apply masking
                if mask3.any():
                    preds3 = outputs['exit3'][mask3].argmax(dim=1)
                    final_preds[mask3] = preds3
                    total_cost += self.cost[2].item() * mask3.sum().item()
                    has_exited |= mask3

                # Exit 4
                mask4 = ~has_exited
                if mask4.any():
                    preds4 = outputs['exit4'][mask4].argmax(dim=1)
                    final_preds[mask4] = preds4
                    total_cost += self.cost[3].item() * mask4.sum().item()
                    has_exited |= mask4
                
                # Tally up
                total_correct += (final_preds == labels).sum().item()
                total_samples += batch_size

        acc = total_correct/total_samples
        avg_cost = total_cost/total_samples
        if isinstance(avg_cost, torch.Tensor):
            avg_cost = avg_cost.item()
        print(f"EM Routing: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": float(acc), "cost": float(avg_cost)}

    def eval_all(self, dataloader, routers, threshold=0.5, branchy_threshold=1.0):
        results = {}
        results['resnet'] = self.eval_resnet(dataloader)
        
        # MultiExit Fixed
        fixed_res = self.eval_multiexit_resnet_fixed(dataloader)
        for k, v in fixed_res.items():
            results[f'fixed_{k}'] = v
            
        results['random'] = self.eval_multiexit_resnet_random(dataloader)
        results['branchynet'] = self.eval_branchynet(dataloader, threshold=branchy_threshold)
        results['em_routing'] = self.eval_em_routing(dataloader, routers=routers, threshold=threshold)
        results['oracle'] = self.eval_oracle(dataloader)
        return results
        
                

                

                

