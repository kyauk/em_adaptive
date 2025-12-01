""""

Goal of this is to save intermediate features after running ResNet-18 Backbone once, so backbone doesn't need to be run multiple times.

"""

import torch
import os
from tqdm import tqdm
from models.multi_exit_resnet import MultiExitResNet

def cache_dataset_features(model, dataloader, save_path):
    model.eval()
    all_features = {
        'layer1': [],
        'layer2': [],
        'layer3': [],
        'layer4': []
    }
    all_labels = []
   #  loop through dataloader -> for each batch, run through model -> save features
    for batch in tqdm(dataloader):
        inputs, labels = batch
        features = model.extract_all_features(inputs)

        for i in range(4):
            all_features[f'layer{i+1}'].append(features[f'layer{i+1}'])
        all_labels.append(labels)
   # concatenate batches into single tesnors 
    final_features = {
        'layer1': torch.cat(all_features['layer1'], dim=0),
        'layer2': torch.cat(all_features['layer2'], dim=0),
        'layer3': torch.cat(all_features['layer3'], dim=0),
        'layer4': torch.cat(all_features['layer4'], dim=0)
    }
    final_labels = torch.cat(all_labels, dim=0)

   # save features and labels to disk
    torch.save({'features': final_features,
            'labels': final_labels},
            save_path)
    print(f"✓ Cached {len(final_labels)} samples to {save_path}")


def load_cached_features(load_path):

    cached_features = torch.load(load_path)
    features = cached_features['features']
    labels = cached_features['labels']
    return features, labels


if __name__ == '__main__':
    # Imports
    from models.multi_exit_resnet import MultiExitResNet
    from dataloader import get_cifar10_loaders
    
    print("=== Feature Caching Script ===\n")
    
    # Create model
    # Create model
    print("Loading model...")
    model = MultiExitResNet(num_classes=10, freeze_backbone=True)
    
    # Load trained backbone weights
    backbone_path = 'checkpoints/backbone/resnet18_cifar10_best.pth'
    if os.path.exists(backbone_path):
        print(f"Loading backbone from {backbone_path}")
        # We need to load state_dict but filter out exit layers if they exist in checkpoint (they shouldn't yet)
        # The backbone training script saves the whole model state_dict
        state_dict = torch.load(backbone_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Backbone checkpoint not found at {backbone_path}. Using random init (bad!)")
    
    # Get dataloaders
    print("Loading CIFAR-10 dataloaders...")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir='./cifar-10-batches-py',
        batch_size=128
    )
    
    # Cache training set
    print("\nCaching training set...")
    cache_dataset_features(model, train_loader, 'cached_features_train.pt')
    
    # Cache test set
    print("\nCaching test set...")
    cache_dataset_features(model, test_loader, 'cached_features_test.pt')
    
    # Verify
    print("\n=== Verification ===")
    train_features, train_labels = load_cached_features('cached_features_train.pt')
    test_features, test_labels = load_cached_features('cached_features_test.pt')
    
    print(f"Train features['layer1']: {train_features['layer1'].shape}")
    print(f"Train labels: {train_labels.shape}")
    print(f"Test features['layer1']: {test_features['layer1'].shape}")
    print(f"Test labels: {test_labels.shape}")
    print("\n✓ Feature caching complete!")
