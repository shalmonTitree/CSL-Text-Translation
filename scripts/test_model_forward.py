import torch
from src.model import CSLTBottleneckModel

# fake batch
B, T, D = 2, 200, 225
feats = torch.randn(B, T, D)
mask = torch.ones(B, T)

model = CSLTBottleneckModel()
model.train()

out = model(feats, mask)

print("Output shape:", out.shape)
