import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
