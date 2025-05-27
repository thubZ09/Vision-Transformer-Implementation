import yaml
import torch
import wandb
from data.dataloader import build_dataloader
from models.vit import build_model
from engine.trainer import train_one_epoch, validate

def main():
    # Load config
    with open("configs/base_vit.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    model = build_model(config)
    dataloaders = build_dataloader(config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['training']['epochs']
    )
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize WandB
    wandb.init(project="vit-colab", config=config)
    
    # Training loop
    for epoch in range(config['training']['epochs']):
        train_metrics = train_one_epoch(
            model, dataloaders['train'], 
            optimizer, scheduler, scaler, epoch, config
        )
        
        val_metrics = validate(model, dataloaders['test'])
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            **train_metrics,
            **val_metrics
        })

if __name__ == "__main__":
    main()