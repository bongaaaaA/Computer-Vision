import torch
import torch.nn as nn

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 64*64)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(labels.size(0), 1, 64, 64)
        x = torch.cat([img, c], dim=1)
        return self.net(x).view(-1)

print("ok ConditionalDiscriminator")