import torch
from torchvision import datasets, transforms
import timm

def build_dataloader(config):
    train_transform = timm.data.create_transform(
        input_size=config['model']['img_size'],
        is_training=True,
        auto_augment=config['data']['augmentation'],
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    test_transform = transforms.Compose([
        transforms.Resize(config['model']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_data = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True,
        transform=train_transform
    )

    test_data = datasets.CIFAR10(
        root='./data',
        train=False,
        transform=test_transform
    )

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_data,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_data,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
    }
    return dataloaders