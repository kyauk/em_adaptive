"""
Train Exit Classifiers

Trains the 4 exit classifiers using cached features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.multi_exit_rnet import MultiExitResNet
from models.exits import ExitClassifier
from algorithms.feature_cache import load_cached_features
import os
import yaml
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def create_dataloaders(features_path, batch_size=128):
    print(f"Loading features from {features_path}...")
    features_dict, labels = load_cached_features(features_path)
    
    dataset = TensorDataset(
        features_dict['layer1'],
        features_dict['layer2'],
        features_dict['layer3'],
        features_dict['layer4'],
        labels
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_exits():
    # Load Config
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Init Model
    print("Initializing model...")
    model = MultiExitResNet(num_classes=10, freeze_backbone=True)
    model.to(device)
    
    # Data Prep
    train_loader = create_dataloaders('cached_features_train.pt', config['dataset']['batch_size'])
    test_loader = create_dataloaders('cached_features_test.pt', config['dataset']['batch_size'])
    
    # Setup Optimizer
    print("Setting up optimizer...")
    # Optimize ONLY exit parameters
    params = list(model.exit1.parameters()) + \
             list(model.exit2.parameters()) + \
             list(model.exit3.parameters()) + \
             list(model.exit4.parameters())
             
    optimizer = optim.SGD(
        params, 
        lr=float(config['training_exits']['lr']),
        momentum=float(config['training_exits']['momentum']),
        weight_decay=float(config['training_exits']['weight_decay'])
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training_exits']['epochs']
    )
    
    # Training Step
    for epoch in range(config['training_exits']['epochs']):
        # verify model is in train mode
        model.train()
        total_loss = 0

        """
        basic DL training pipeline goes:
        zero out the gradients to get rid of last iter's grads
        1. forward pass to get score and loss
        2. backward pass 
        3. optimize
        """
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for f1, f2, f3, f4, labels in pbar:
            f1, f2, f3, f4 = f1.to(device), f2.to(device), f3.to(device), f4.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # forward pass
            out1 = model.exit1(f1)
            out2 = model.exit2(f2)
            out3 = model.exit3(f3)
            out4 = model.exit4(f4)
            
            # compute total loss
            loss = criterion(out1, labels) + criterion(out2, labels) + criterion(out3, labels) + criterion(out4, labels) 
            loss.backward()
            optimizer.step()
            
            # updating progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
        
        # scheduler step
        scheduler.step()
        
        # checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_dir = config['training_exits']['save_dir']
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'exits_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
        
            # Evaluate
            evaluate(model, test_loader, device)

    print("\nTraining complete!")
    # Save final model
    save_dir = config['training_exits']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'exits_final.pth'))

def evaluate(model, dataloader, device):
    model.eval()
    correct = [0, 0, 0, 0]
    total = 0
    
    with torch.no_grad():
        for f1, f2, f3, f4, labels in dataloader:
            f1, f2, f3, f4 = f1.to(device), f2.to(device), f3.to(device), f4.to(device)
            labels = labels.to(device)
            
            # Get predictions from each exit
            outs = [
                model.exit1(f1),
                model.exit2(f2),
                model.exit3(f3),
                model.exit4(f4)
            ]
            
            total += labels.size(0)
            for i, out in enumerate(outs):
                pred = out.argmax(dim=1)
                correct[i] += (pred == labels).sum().item()
    
    print("\nValidation Accuracy:")
    for i in range(4):
        acc = 100 * correct[i] / total
        print(f"Exit {i+1}: {acc:.2f}%")




if __name__ == '__main__':
    train_exits()
