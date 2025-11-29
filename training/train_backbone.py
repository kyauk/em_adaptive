import torch
import torch.nn as nn
import torch.optim as optim
from models.multi_exit_rnet import MultiExitResNet
from dataloader import get_cifar10_loaders
import os
from tqdm import tqdm

def train_backbone():
    # Config
    EPOCHS = 30 # Should be enough to reach >90% with this architecture
    BATCH_SIZE = 128
    LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    SAVE_DIR = 'checkpoints/backbone'
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE)
    
    # Model - Initialize with freeze_backbone=False to train it!
    print("Initializing model...")
    model = MultiExitResNet(num_classes=10, freeze_backbone=False)
    model.to(device)
    
    # We only care about the final exit for backbone training
    # But the model forward returns exit4 by default if return_all_exits=False
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    best_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Returns exit4 logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss/(pbar.n+1), 'acc': 100.*correct/total})
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'resnet18_cifar10_best.pth'))
            print(f"Saved best model ({best_acc:.2f}%)")
            
    print("Training complete!")

if __name__ == '__main__':
    train_backbone()
