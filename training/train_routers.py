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

def train_routers(lambda_val=0.05):
    EPS = 1e-8
    EPOCHS = 50  # Increase for better router convergence
    TRAIN_FEATURES_PATH = "cached_features_train.pt"
    BATCH_SIZE = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = MultiExitResNet()
    if os.path.exists("checkpoints/exits/exits_final.pth"):
        state_dict = torch.load("checkpoints/exits/exits_final.pth", map_location=device)
        model_state = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded checkpoint with {len(filtered_state_dict)}/{len(state_dict)} matched keys.")
    else:
        print("Warning: No exit checkpoint found. Using random initialization for exits (this might be bad for EM).")
    
    model.to(device)

    # Load Data
    print("Loading cached features...")
    if not os.path.exists(TRAIN_FEATURES_PATH):
        print(f"Error: {TRAIN_FEATURES_PATH} not found. Please run feature caching first.")
        return

    features_dict, labels = load_cached_features(TRAIN_FEATURES_PATH)
    f1 = features_dict['layer1']
    f2 = features_dict['layer2']
    f3 = features_dict['layer3']
    f4 = features_dict['layer4']

    # Create dataset for EM (needs to be on device or moved during loop)
    dataset = torch.utils.data.TensorDataset(f1, f2, f3, f4, labels)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Generate EM Assignments
    print(f"Running EM to generate targets (Lambda={lambda_val})...")
    em = EMRouting(model, lambda_val=lambda_val)
    assignments, all_labels = em.run(train_loader)

    # Initialize Routers
    print(f"Initializing Routers (Linear Logits)...")
    router1 = Router(input_dim=f1.shape[1]).to(device)
    router2 = Router(input_dim=f2.shape[1]).to(device)
    router3 = Router(input_dim=f3.shape[1]).to(device)
    routers = [router1, router2, router3]

    # Define Optimizers
    optimizers = [optim.Adam(r.parameters(), lr=0.001) for r in routers]
    
    # BCEWithLogitsLoss handles sigmoid + BCE numerically stably
    criterion = nn.BCEWithLogitsLoss()

    # Create New Dataset for Routers
    router_dataset = torch.utils.data.TensorDataset(f1, f2, f3, assignments)
    router_loader = DataLoader(router_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training Loop for Routers
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for batch in router_loader:
            bf1, bf2, bf3, b_assignments = batch
            bf1, bf2, bf3 = bf1.to(device), bf2.to(device), bf3.to(device)
            b_assignments = b_assignments.to(device)
            
            # Router 1
            target1 = b_assignments[:, 0:1] 
            router1.train()
            optimizers[0].zero_grad()
            pred1 = router1(bf1)
            loss1 = criterion(pred1, target1)
            loss1.backward()
            optimizers[0].step()
            
            # Router 2
            # p(e2, given not p1) = assignment_2 / (assignments_2 + assignments_3 + assignments_4)
            target2 = b_assignments[:, 1:2] / (b_assignments[:, 1:].sum(dim=1, keepdim=True) + EPS)
            router2.train()
            optimizers[1].zero_grad()
            pred2 = router2(bf2)
            loss2 = criterion(pred2, target2)
            loss2.backward()
            optimizers[1].step()

            # Router 3
            # p(e3 | not e1, not e2) = assignment_3 / (assignments_3 + assignments_4)
            target3 = b_assignments[:, 2:3] / (b_assignments[:, 2:].sum(dim=1, keepdim=True) + EPS)   
            router3.train()
            optimizers[2].zero_grad()
            pred3 = router3(bf3)
            loss3 = criterion(pred3, target3)
            loss3.backward()
            optimizers[2].step()
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'router1': router1.state_dict(),
        'router2': router2.state_dict(),
        'router3': router3.state_dict()
    }, 'checkpoints/routers.pth')
    print("Routers saved to checkpoints/routers.pth")

if __name__ == "__main__":
    train_routers()