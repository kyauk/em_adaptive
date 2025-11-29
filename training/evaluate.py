import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.multi_exit_rnet import MultiExitResNet
from models.exits import ExitClassifier
from algorithms.feature_cache import load_cached_features
import os
import sys

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def create_dataloader(features_path, batch_size=128):
    print(f"Loading features from {features_path}...")
    features_dict, labels = load_cached_features(features_path)
    
    dataset = TensorDataset(
        features_dict['layer1'],
        features_dict['layer2'],
        features_dict['layer3'],
        features_dict['layer4'],
        labels
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate_model():
    device = get_device()
    print(f"Using device: {device}")
    
    # Init Model
    print("Initializing model...")
    model = MultiExitResNet(num_classes=10, freeze_backbone=True)
    
    # Load Backbone
    backbone_path = 'checkpoints/backbone/resnet18_cifar10_best.pth'
    if os.path.exists(backbone_path):
        print(f"Loading backbone from {backbone_path}")
        state_dict = torch.load(backbone_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Error: Backbone checkpoint not found at {backbone_path}")
        return
        
    model.to(device)
    model.eval()
    
    # Initialize Exits
    exit1 = ExitClassifier(64, 10).to(device)
    exit2 = ExitClassifier(128, 10).to(device)
    exit3 = ExitClassifier(256, 10).to(device)
    exit4 = ExitClassifier(512, 10).to(device)
    
    exits = [exit1, exit2, exit3, exit4]
    
    # Load Exits
    exits_path = 'checkpoints/exits/exits_final.pth'
    if not os.path.exists(exits_path):
         # Try old path
         exits_path = 'checkpoints/exits_final.pth'
         
    if os.path.exists(exits_path):
        print(f"Loading exits from {exits_path}")
        checkpoint = torch.load(exits_path, map_location=device)
        exit1.load_state_dict(checkpoint['exit1'])
        exit2.load_state_dict(checkpoint['exit2'])
        exit3.load_state_dict(checkpoint['exit3'])
        exit4.load_state_dict(checkpoint['exit4'])
    else:
        print(f"Error: Exits checkpoint not found at {exits_path}")
        return
        
    for exit in exits:
        exit.eval()
    
    # Data Prep
    features_path = 'cached_features_test.pt'
    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found. Please run algorithms/feature_cache.py first.")
        return
        
    test_loader = create_dataloader(features_path, batch_size=128)
    
    # Evaluation Loop
    correct = [0] * 4
    total = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for f1, f2, f3, f4, labels in test_loader:
            f1, f2, f3, f4 = f1.to(device), f2.to(device), f3.to(device), f4.to(device)
            labels = labels.to(device)
            
            outs = [
                exit1(f1),
                exit2(f2),
                exit3(f3),
                exit4(f4)
            ]
            
            total += labels.size(0)
            for i, out in enumerate(outs):
                pred = out.argmax(dim=1)
                correct[i] += (pred == labels).sum().item()
    
    print("\nTest Set Accuracy:")
    for i in range(4):
        acc = 100 * correct[i] / total
        print(f"Exit {i+1}: {acc:.2f}%")

if __name__ == '__main__':
    evaluate_model()
