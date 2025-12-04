"""Quick check of router outputs"""
import torch
from models.routers import Router
from algorithms.feature_cache import load_cached_features

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

# Load routers
router_ckpt = torch.load("checkpoints/routers.pth", map_location=device)
router1 = Router(input_dim=64, use_mlp=True).to(device)
router2 = Router(input_dim=128, use_mlp=True).to(device)
router3 = Router(input_dim=256, use_mlp=True).to(device)
router1.load_state_dict(router_ckpt['router1'])
router2.load_state_dict(router_ckpt['router2'])
router3.load_state_dict(router_ckpt['router3'])
router1.eval()
router2.eval()
router3.eval()

# Load cached features
features, labels = load_cached_features('cached_features_train.pt')

# Test on first 1000 samples
f1 = features['layer1'][:1000].to(device)
f2 = features['layer2'][:1000].to(device)
f3 = features['layer3'][:1000].to(device)

with torch.no_grad():
    p1 = router1(f1).squeeze()
    p2 = router2(f2).squeeze()
    p3 = router3(f3).squeeze()

print(f"\nRouter 1 outputs: min={p1.min():.4f}, max={p1.max():.4f}, mean={p1.mean():.4f}")
print(f"Router 2 outputs: min={p2.min():.4f}, max={p2.max():.4f}, mean={p2.mean():.4f}")
print(f"Router 3 outputs: min={p3.min():.4f}, max={p3.max():.4f}, mean={p3.mean():.4f}")

print(f"\n% that would exit at threshold 0.5:")
print(f"  Router 1: {(p1 > 0.5).float().mean() * 100:.1f}%")
print(f"  Router 2: {(p2 > 0.5).float().mean() * 100:.1f}%")
print(f"  Router 3: {(p3 > 0.5).float().mean() * 100:.1f}%")

print(f"\n% that would exit at threshold 0.3:")
print(f"  Router 1: {(p1 > 0.3).float().mean() * 100:.1f}%")
print(f"  Router 2: {(p2 > 0.3).float().mean() * 100:.1f}%")
print(f"  Router 3: {(p3 > 0.3).float().mean() * 100:.1f}%")
