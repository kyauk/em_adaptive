import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.routers import Router
from algorithms.em_routing import EMRouting
from algorithms.feature_cache import load_cached_features
from models.multi_exit_resnet import MultiExitResNet

EPOCHS = 20
TRAIN_FEATURES_PATH = "cached_features_train.pt"
BATCH_SIZE = 128
USE_MLP = True  # Toggle: True for MLP, False for Linear
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = MultiExitResNet()
# Load model with strict=False to handle potential backbone mismatches (7x7 vs 3x3)
# We only care about the Exit Classifiers being correct.
state_dict = torch.load("checkpoints/exits_final.pth")
model_state = model.state_dict()
# Filter out mismatching keys (like conv1.weight)
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
model.load_state_dict(filtered_state_dict, strict=False)
model.to(device)
print(f"Loaded checkpoint with {len(filtered_state_dict)}/{len(state_dict)} matched keys.")

# Load Data
print("Loading cached features...")
features_dict, labels = load_cached_features(TRAIN_FEATURES_PATH)
f1 = features_dict['layer1']
f2 = features_dict['layer2']
f3 = features_dict['layer3']
f4 = features_dict['layer4']

# Create dataset for EM (needs to be on device or moved during loop)
# Note: TensorDataset keeps data on CPU by default until moved
dataset = torch.utils.data.TensorDataset(f1, f2, f3, f4, labels)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Generate EM Assignments
print("Running EM to generate targets...")
em = EMRouting(model)
assignments, all_labels = em.run(train_loader)

# Compute Hard Assignments
hard_assignments = torch.argmax(assignments, dim=1)
print(f"EM Assignments Distribution: {torch.bincount(hard_assignments)}")

# Initialize Routers
print(f"Initializing Routers (USE_MLP={USE_MLP})...")
router1 = Router(input_dim=f1.shape[1], use_mlp=USE_MLP).to(device)
router2 = Router(input_dim=f2.shape[1], use_mlp=USE_MLP).to(device)
router3 = Router(input_dim=f3.shape[1], use_mlp=USE_MLP).to(device)
routers = [router1, router2, router3]

# Define Optimizers and Criterion
optimizers = [optim.Adam(r.parameters(), lr=0.001) for r in routers]
# Binary cross-entropy loss since we're working with binary classification
criterion = nn.BCELoss()
# Create New Dataset for Routers
router_dataset = torch.utils.data.TensorDataset(f1, f2, f3, hard_assignments)
router_loader = DataLoader(router_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training Loop for Routers
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for batch in router_loader:
        bf1, bf2, bf3, b_labels = batch
        bf1, bf2, bf3 = bf1.to(device), bf2.to(device), bf3.to(device)
        b_labels = b_labels.to(device)
        # training router 1
        target1 = (b_labels==0).float().unsqueeze(1)
        router1.train()
        optimizers[0].zero_grad()
        pred1 = router1(bf1)
        loss1 = criterion(pred1, target1)
        loss1.backward()
        optimizers[0].step()
        # training router 2
        target2 = (b_labels==1).float().unsqueeze(1)
        router2.train()
        optimizers[1].zero_grad()
        pred2 = router2(bf2)
        loss2 = criterion(pred2, target2)
        loss2.backward()
        optimizers[1].step()
        # training router 3
        target3 = (b_labels==2).float().unsqueeze(1)
        router3.train()
        optimizers[2].zero_grad()
        pred3 = router3(bf3)
        loss3 = criterion(pred3, target3)
        loss3.backward()
        optimizers[2].step()
        
        
    print(f"Epoch {epoch+1}/{EPOCHS} complete.")

# Save Routers
os.makedirs('checkpoints', exist_ok=True)
torch.save({
    'router1': router1.state_dict(),
    'router2': router2.state_dict(),
    'router3': router3.state_dict()
}, 'checkpoints/routers.pth')
print("Routers saved to checkpoints/routers.pth")