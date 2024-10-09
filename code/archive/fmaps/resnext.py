import torch

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

for name, layer in model.named_modules():
    print(name)