import torch
from tqdm import tqdm
import wandb

def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, config):
    model.train()
    total_loss = 0
    correct = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
    for inputs, targets in pbar:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        scaler.scale(loss).backward()
        if config['training']['clip_grad'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        
        pbar.set_postfix({'Loss': loss.item()})
    
    scheduler.step()
    
    return {
        'train_loss': total_loss / len(loader),
        'train_acc': correct / len(loader.dataset)
    }

def validate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
    
    return {
        'val_loss': total_loss / len(loader),
        'val_acc': correct / len(loader.dataset)
    }