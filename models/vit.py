import torch
import timm
from torch import nn

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = timm.create_model(
            config['model']['name'],
            img_size=config['model']['img_size'],
            patch_size=config['model']['patch_size'],
            in_chans=config['model']['in_chans'],
            embed_dim=config['model']['embed_dim'],
            depth=config['model']['depth'],
            num_heads=config['model']['num_heads'],
            mlp_ratio=config['model']['mlp_ratio'],
            qkv_bias=config['model']['qkv_bias'],
            drop_rate=config['model']['drop_rate'],
            attn_drop_rate=config['model']['attn_drop_rate'],
            drop_path_rate=config['model']['drop_path_rate'],
            num_classes=10  # For CIFAR10
        )
        
    def forward(self, x):
        return self.vit(x)

def build_model(config):
    return VisionTransformer(config).cuda()