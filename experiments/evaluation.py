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
        print(f"Standard ResNet-18: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": acc, "cost": avg_cost}

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
            results[exit_name] = {"accuracy": acc, "cost": avg_cost}
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
        print(f"Random Routing: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": acc, "cost": avg_cost}

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
        print(f"BranchyNet: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": acc, "cost": avg_cost}

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
                
                batch_size = images.size(0)
                sample_costs = torch.zeros(batch_size).to(self.device)
                # Default to incorrect and max cost
                sample_correct = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
                sample_costs[:] = self.cost[3] 
                
                # Check exits in order
                for i in range(4):
                    exit_name = f"exit{i+1}"
                    preds = outputs[exit_name].argmax(dim=1)
                    is_correct = (preds == labels)
                    
                    # If correct and haven't found a correct exit yet, take this one (update if is_correct is True AND sample_correct is False)
                    update_mask = is_correct * (sample_correct == 0)
                    total_cost += self.cost[i] * update_mask.sum().item()
                    sample_correct += update_mask
                
                total_samples += batch_size
                total_correct += sample_correct.sum().item()

        acc = total_correct/total_samples
        avg_cost = total_cost/total_samples
        print(f"Oracle: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": acc, "cost": avg_cost}

    def eval_em_routing(self, dataloader, routers, threshold=0.5):
        """
        EM Routing strategy: Use trained routers to decide when to exit.
        routers: list of 4 router models (or dict)
        threshold: Probability threshold for exiting (default: 0.5)
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_cost = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                # return all exits = True because we need to evaluate accuracy for each exit
                # outputs are [num_exits, B, num_classes]
                outputs, features = self.model(images, return_all_exits=True, return_features=True)
                
                # Get features from all 4 layers, features are [B, C, H, W]
                f1 = features['feature1']
                f2 = features['feature2']
                f3 = features['feature3']
                f4 = features['feature4']
                final_preds = torch.zeros(images.size(0), dtype=torch.long).to(self.device)
                
                # run images through router 1 (example output is [0.1, 0.5, 0.8, 0.6...])
                p1 = routers[0](f1)
                # mask 
                mask1 = (p1 > threshold).int().squeeze()
                # add cost to total cost
                total_cost += self.cost[0] * mask1.sum().item()
                # get predictions from outputs to which exit to take
                preds1 = outputs['exit1'][mask1==1].argmax(dim=1)

                # run images through router 2
                p2 = routers[1](f2)
                
                # mask, make sure to remove items that's been predicted by router 1
                mask2 = (p2 > threshold).int().squeeze()
                mask2 = mask2 * (mask1 == 0).int()
                # add cost to total cost
                total_cost += self.cost[1] * mask2.sum().item()
                # get predictions from outputs to which exit to take
                preds2 = outputs['exit2'][mask2==1].argmax(dim=1)

                # run images through router 3
                p3 = routers[2](f3)
                # mask, make sure to remove items that's been predicted by router 1 and 2
                mask3 = (p3 > threshold).int().squeeze()
                mask3 = mask3 * (mask1 == 0).int() * (mask2 == 0).int()
                # add cost to total cost
                total_cost += self.cost[2] * mask3.sum().item()
                # get predictions from outputs to which exit to take
                preds3 = outputs['exit3'][mask3==1].argmax(dim=1)

                # if no router predicted, use last exit
                mask4 = (mask1 == 0).int() * (mask2 == 0).int() * (mask3 == 0).int()
                # add cost to total cost
                total_cost += self.cost[3] * mask4.sum().item()
                # get predictions from outputs to which exit to take
                preds4 = outputs['exit4'][mask4==1].argmax(dim=1)
                
                # update final predictions
                final_preds[mask1==1] = preds1
                final_preds[mask2==1] = preds2
                final_preds[mask3==1] = preds3
                final_preds[mask4==1] = preds4

                total_correct += (final_preds == labels).sum().item()
                total_samples += final_preds.size(0)

        acc = total_correct/total_samples
        avg_cost = total_cost/total_samples
        print(f"EM Routing: Accuracy={acc:.4f}, Cost={avg_cost:.4f}")
        return {"accuracy": acc, "cost": avg_cost}

    def eval_all(self, dataloader, routers, threshold=0.5):
        self.eval_resnet(dataloader)
        self.eval_multiexit_resnet_fixed(dataloader)
        self.eval_multiexit_resnet_random(dataloader)
        self.eval_branchynet(dataloader, threshold=threshold)
        self.eval_em_routing(dataloader, routers=routers, threshold=threshold)
        self.eval_oracle(dataloader)
        return
        
                

                

                

