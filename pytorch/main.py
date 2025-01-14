import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from vit import VisionTransformer
from dataset import get_dataloaders
from utils import calculate_accuracy

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    total_accuracy = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_accuracy += calculate_accuracy(outputs, labels)
    
    return running_loss / len(dataloader), total_accuracy / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)

    return running_loss / len(dataloader), total_accuracy / len(dataloader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(img_size=224, patch_size=16, num_classes=10).to(device)
    train_loader, test_loader = get_dataloaders(batch_size=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
