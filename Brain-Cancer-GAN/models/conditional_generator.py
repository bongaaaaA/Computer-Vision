import torch
import torch.nn as nn

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=2):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, z_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim*2, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, c], dim=1)
        return self.net(x)

print("ok ConditionalGenerator")