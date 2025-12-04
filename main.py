import argparse
import torch
import os
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.multi_exit_resnet import MultiExitResNet
from models.routers import Router
from training.train_exits import train_exits
from training.train_routers import train_routers
from algorithms.em_routing import EMRouting
from experiments.evaluation import Evaluator

def get_dataloader(batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = datasets.CIFAR10(root='./cifar-10-batches-py', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)

def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="EM Adaptive Computation")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train_exits", "run_em", "train_routers", "evaluate"],
                        help="Action to perform")
    parser.add_argument("--method", type=str, default="all",
                        choices=["resnet", "fixed", "random", "branchynet", "oracle", "em", "all"],
                        help="Evaluation method")
    parser.add_argument("--lambda_val", type=float, default=0.5, help="Lambda for EM routing")
    parser.add_argument("--threshold", type=float, default=0.5, help="Entropy threshold for BranchyNet")
    
    args = parser.parse_args()
    config = load_config()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Initialize Model
    model = MultiExitResNet(num_classes=10, freeze_backbone=True).to(device)
    
    # 1. Load Backbone
    backbone_path = "checkpoints/backbone/resnet18_cifar10_best.pth"
    if os.path.exists(backbone_path):
        print(f"Loading backbone from {backbone_path}...")
        # Load with strict=False because backbone checkpoint won't have exits
        try:
            model.load_state_dict(torch.load(backbone_path, map_location=device), strict=False)
        except Exception as e:
            print(f"Warning: Failed to load backbone: {e}")
    else:
        print("Warning: Backbone checkpoint not found!")

    # 2. Load Exits
    # Only load if we are not training them (or if we want to continue training? usually we start fresh or load)
    # But for evaluation we MUST load them.
    if args.mode != "train_exits" and os.path.exists("checkpoints/exits_final.pth"):
        print("Loading trained exit classifiers...")
        exit_state = torch.load("checkpoints/exits_final.pth", map_location=device)
        
        # Check if it's a nested dict (e.g. {'exit1': state_dict, ...})
        if 'exit1' in exit_state and isinstance(exit_state['exit1'], dict):
            print("Detected nested exit checkpoint format.")
            model.exit1.load_state_dict(exit_state['exit1'])
            model.exit2.load_state_dict(exit_state['exit2'])
            model.exit3.load_state_dict(exit_state['exit3'])
            model.exit4.load_state_dict(exit_state['exit4'])
        else:
            # Try standard load
            try:
                model.load_state_dict(exit_state, strict=False)
            except Exception as e:
                print(f"Error loading exits: {e}")
    
    if args.mode == "train_exits":
        print("Training Exit Classifiers...")
        train_exits()

    elif args.mode == "run_em":
        print(f"Running EM Algorithm (Lambda={args.lambda_val})...")
        # Load cached features
        features_path = "cached_features_train.pt"
        if not os.path.exists(features_path):
            print("Cached features not found. Please run feature caching first.")
            return
            
        # Load data
        # We need to load data to pass to EM.run()
        # But EMRouting.run() takes a dataloader.
        # So we need to create a dataloader from cached features.
        from algorithms.feature_cache import load_cached_features
        features_dict, labels = load_cached_features(features_path)
        dataset = torch.utils.data.TensorDataset(
            features_dict['layer1'], features_dict['layer2'], 
            features_dict['layer3'], features_dict['layer4'], labels
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        em = EMRouting(model, lambda_val=args.lambda_val)
        em.run(loader, iterations=10)

    elif args.mode == "train_routers":
        print("Training Routers...")
        train_routers()

    elif args.mode == "evaluate":
        print(f"Evaluating Method: {args.method}")
        test_loader = get_dataloader(train=False)
        evaluator = Evaluator(model)
        
        if args.method in ["resnet", "all"]:
            evaluator.eval_resnet(test_loader)
            
        if args.method in ["fixed", "all"]:
            evaluator.eval_multiexit_resnet_fixed(test_loader)
            
        if args.method in ["random", "all"]:
            evaluator.eval_multiexit_resnet_random(test_loader)
            
        if args.method in ["branchynet", "all"]:
            evaluator.eval_branchynet(test_loader, threshold=args.threshold)
            
        if args.method in ["oracle", "all"]:
            evaluator.eval_oracle(test_loader)
            
        if args.method in ["em", "all"]:
            # Load routers
            routers = []
            # Checkpoint path for routers
            router_path = "checkpoints/routers.pth"
            if os.path.exists(router_path):
                router_state = torch.load(router_path, map_location=device)
                # Assuming router_state is a list of state_dicts or a dict of state_dicts
                # We need to know the structure.
                # We need to know the structure.
                # We instantiate 3 routers (for exits 1, 2, 3).
                in_features = [64, 128, 256]
                for i in range(3):
                    r = Router(in_features[i], hidden_dim=64).to(device)
                    routers.append(r)
                
                # Load weights properly
                if isinstance(router_state, dict):
                    for i in range(3):
                        key = f"router{i+1}"
                        if key in router_state:
                            routers[i].load_state_dict(router_state[key])
                
                evaluator.eval_em_routing(test_loader, routers, threshold=args.threshold)
            else:
                print("Router checkpoint not found. Skipping EM evaluation.")   

if __name__ == "__main__":
    main()