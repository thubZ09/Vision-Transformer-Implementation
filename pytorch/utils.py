import torch

def calculate_accuracy(predictions, labels):
    _, preds = torch.max(predictions, 1)
    return (preds == labels).float().mean().item()
