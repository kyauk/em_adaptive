"""
Main script for running EM Adaptive Computation.
"""


import argparse
import torch
import os
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataloader import get_cifar10_loaders

# Import models, training scripts, algorithms, and evals 
from models.multi_exit_resnet import MultiExitResNet
from models.routers import Router
from training.train_exits import train_exits
from training.train_routers import train_routers
from algorithms.em_routing import EMRouting
from experiments.evaluation import Evaluator

# configuration (device, path)
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    CHECKPOINT_DIR = "checkpoints"
    BACKBONE_PATH = os.path.join(CHECKPOINT_DIR, "backbone/resnet18_cifar10_best.pth")
    ROUTER_PATH = os.path.join(CHECKPOINT_DIR, "routers.pth")
    EXIT_PATH = os.path.join(CHECKPOINT_DIR, "exits/exits_final.pth")
    CACHED_TRAIN_FEATURES_PATH = "cached_features_train.pth"
    CACHED_TEST_FEATURES_PATH = "cached_features_test.pth"
    
    # Data
    DATA_DIR = "./cifar-10-batches-py"
    BATCH_SIZE = 128
    # note to self: make this changeable with argparse later
    NUM_WORKERS = 2
    
    # Model
    NUM_CLASSES = 10
    ROUTER_HIDDEN_DIM = 64
    

    
# dataloader
def get_dataloader():
    return get_cifar10_loaders(
        data_dir=Config.DATA_DIR,
        batch_size=Config.BATCH_SIZE, 
        num_workers=Config.NUM_WORKERS,
    )
# setup the model
def setup_model(device=Config.DEVICE, load_backbone=True, load_exits=False):
    model = MultiExitResNet(num_classes=Config.NUM_CLASSES, freeze_backbone=True).to(Config.DEVICE)
    if load_backbone:
        if os.path.exists(Config.BACKBONE_PATH):
            model.load_state_dict(torch.load(Config.BACKBONE_PATH, map_location=device), strict=False)
            print("Backbone loaded!")
        else:
            print(f"Error: Backbone not found at {Config.BACKBONE_PATH}")    
    if load_exits:
        if os.path.exists(Config.EXIT_PATH):
            state = torch.load(Config.EXIT_PATH, map_location=device)
            # Handle the Nested Dict case:
            if 'exit1' in state:
                model.exit1.load_state_dict(state['exit1'])
                model.exit2.load_state_dict(state['exit2'])
                model.exit3.load_state_dict(state['exit3'])
                model.exit4.load_state_dict(state['exit4'])
            else:
                model.load_state_dict(state)
            print("Exits loaded!")
        else:
            print(f"Error: Exits not found at {Config.EXIT_PATH}")
    return model

def setup_routers(device=Config.DEVICE, load_routers=False):
    routers = [
        Router(input_dim=64, hidden_dim=Config.ROUTER_HIDDEN_DIM).to(device),
        Router(input_dim=128, hidden_dim=Config.ROUTER_HIDDEN_DIM).to(device),
        Router(input_dim=256, hidden_dim=Config.ROUTER_HIDDEN_DIM).to(device)
    ]

    if load_routers:
        if os.path.exists(Config.ROUTER_PATH):
            state = torch.load(Config.ROUTER_PATH, map_location=device)
            routers[0].load_state_dict(state['router1'])
            routers[1].load_state_dict(state['router2'])
            routers[2].load_state_dict(state['router3'])
            print("Routers loaded!")
        else:
            print(f"Error: Routers not found at {Config.ROUTER_PATH}")
    return routers

# training
def train_models(mode, lambda_val=0.05):
    if mode == "train_exits" or mode == "train_all":
        print("Training Exits...")
        train_exits()
        print("Exits trained!")
    if mode == "train_routers" or mode == "train_all":
        print(f"Training Routers (Lambda = {lambda_val})...")
        train_routers(lambda_val=lambda_val)
        print("Routers trained!")

# evaluation
def evaluate_models(method, threshold=0.5):
    model = setup_model(load_exits = True)
    evaluator = Evaluator(model)
    _, test_loader = get_dataloader()
    routers = None
    if method in ["em_routing", "all"]:
        routers = setup_routers(load_routers=True)
    if method == "resnet":
        return evaluator.eval_resnet(test_loader)
    elif method == "multiexit_fixed":
        return evaluator.eval_multiexit_resnet_fixed(test_loader)
    elif method == "multiexit_random":
        return evaluator.eval_multiexit_resnet_random(test_loader)
    elif method == "branchynet":
        return evaluator.eval_branchynet(test_loader, threshold=threshold)
    elif method == "em_routing":
        return evaluator.eval_em_routing(test_loader, routers=routers, threshold=threshold)
    elif method == "oracle":
        return evaluator.eval_oracle(test_loader)
    elif method == "all":
        return evaluator.eval_all(test_loader, routers=routers, threshold=threshold)
    else:
        raise ValueError(f"Invalid method: {method}")
    

def main():
    parser = argparse.ArgumentParser(description="EM Adaptive Computation Orchestrator")
    
    # Top-level mode
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["train", "evaluate"],
                        help="Main execution mode: 'train' or 'evaluate'")
    
    # Training arguments
    parser.add_argument("--train_target", type=str, default="train_all",
                        choices=["train_exits", "train_routers", "train_all"],
                        help="What to train (only used if mode='train')")
    
    # Evaluation arguments
    parser.add_argument("--method", type=str, default="all",
                        choices=["resnet", "multiexit_fixed", "multiexit_random", 
                                 "branchynet", "em_routing", "oracle", "all"],
                        help="Evaluation method (only used if mode='evaluate')")
    
    # Hyperparameters
    parser.add_argument("--lambda_val", type=float, default=0.05, 
                        help="Lambda for EM routing (trade-off parameter)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Entropy/Probability threshold for early exiting")
    
    args = parser.parse_args()
    
    print(f"---- EM Adaptive Computation ----")
    print(f"Mode: {args.mode}")
    print(f"Device: {Config.DEVICE}")
    
    if args.mode == "train":
        print(f"Training Target: {args.train_target}")
        train_models(args.train_target, lambda_val=args.lambda_val)
        
    elif args.mode == "evaluate":
        print(f"Evaluation Method: {args.method}")
        evaluate_models(args.method, threshold=args.threshold)
if __name__ == "__main__":
    main()