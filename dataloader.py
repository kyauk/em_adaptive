"""
CIFAR-10 Data Loader
Loads CIFAR-10 from local pickle files and provides PyTorch DataLoaders
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10Dataset(Dataset):
    """Custom CIFAR-10 Dataset loading from pickle files"""
    
    def __init__(self, data_dir, train=True, transform=None):
        """
        Args:
            data_dir: Path to cifar-10-batches-py directory
            train: If True, load training data; else load test data
            transform: Torchvision transforms to apply
        """
        self.transform = transform
        self.data = []
        self.labels = []
        
        if train:
            # Load all training batches
            for i in range(1, 6):
                batch_file = os.path.join(data_dir, f'data_batch_{i}')
                with open(batch_file, 'rb') as f:
                    batch_dict = pickle.load(f, encoding='bytes')
                    self.data.append(batch_dict[b'data'])
                    self.labels.extend(batch_dict[b'labels'])
            
            # Concatenate all batches
            self.data = np.vstack(self.data)
        else:
            # Load test batch
            test_file = os.path.join(data_dir, 'test_batch')
            with open(test_file, 'rb') as f:
                test_dict = pickle.load(f, encoding='bytes')
                self.data = test_dict[b'data']
                self.labels = test_dict[b'labels']
        
        # Reshape data from (N, 3072) to (N, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32)
        # Convert to HWC format (Height, Width, Channels)
        self.data = self.data.transpose(0, 2, 3, 1)
        
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # Apply transforms if specified
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_cifar10_loaders(data_dir='./cifar-10-batches-py', batch_size=128, 
                        num_workers=4, pin_memory=True):
    """
    Get CIFAR-10 train and test data loaders with proper preprocessing
    
    Args:
        data_dir: Path to cifar-10-batches-py directory
        batch_size: Batch size for training and testing
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, test_loader
    """
    
    # CIFAR-10 normalization statistics
    # Computed from training set: mean and std per channel
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Test transforms
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Create datasets
    train_dataset = CIFAR10Dataset(
        data_dir=data_dir,
        train=True,
        transform=train_transform
    )
    
    test_dataset = CIFAR10Dataset(
        data_dir=data_dir,
        train=False,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader


# Class names for CIFAR-10
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_class_names():
    return CIFAR10_CLASSES


if __name__ == '__main__':
    # Test the data loader
    print("Testing CIFAR-10 DataLoader...")
    
    train_loader, test_loader = get_cifar10_loaders(
        data_dir='./cifar-10-batches-py',
        batch_size=128
    )
    
    print(f"Training set: {len(train_loader.dataset)} samples")
    print(f"Test set: {len(test_loader.dataset)} samples")
    print(f"Number of batches (train): {len(train_loader)}")
    print(f"Number of batches (test): {len(test_loader)}")
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Sample labels: {labels[:10].tolist()}")
    print(f"\nClass names: {CIFAR10_CLASSES}")
    
    print("\nSuccess: DataLoader working correctly!")
