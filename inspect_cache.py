import torch

def inspect(path):
    print(f"Inspecting {path}...")
    data = torch.load(path)
    features = data['features']
    labels = data['labels']
    
    print(f"Labels shape: {labels.shape}")
    print(f"Labels sample: {labels[:10]}")
    
    for k, v in features.items():
        print(f"{k}: shape={v.shape}, mean={v.mean():.4f}, std={v.std():.4f}, min={v.min():.4f}, max={v.max():.4f}")
        if torch.isnan(v).any():
            print(f"WARNING: NaNs found in {k}")
        if (v == 0).all():
            print(f"WARNING: All zeros in {k}")

inspect('cached_features_train.pt')
inspect('cached_features_test.pt')
