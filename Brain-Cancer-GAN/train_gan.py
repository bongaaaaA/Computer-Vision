import torch
import torch.nn as nn
from torch.optim import Adam
from utils.data_loader import get_dataloaders
from models.conditional_generator import ConditionalGenerator
from models.conditional_discriminator import ConditionalDiscriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_loader, test_loader = get_dataloaders("C:\\Users\\ALSHELSS\\OneDrive\\Desktop\\Brain-Cancer-GAN\\Computer-Vision\\Brain-Cancer-GAN\\data\\chest_xray")

z_dim = 100
num_classes = 2
G = ConditionalGenerator(z_dim, num_classes).to(device)
D = ConditionalDiscriminator(num_classes).to(device)

criterion = nn.BCELoss()
opt_G = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(50):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.size(0)
        
        # Train Discriminator
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = G(noise, labels)
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)
        
        loss_real = criterion(D(imgs, labels), real_labels)
        loss_fake = criterion(D(fake_imgs.detach(), labels), fake_labels)
        loss_D = loss_real + loss_fake
        
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()
        
        # Train Generator
        loss_G = criterion(D(fake_imgs, labels), real_labels)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
    
    print(f"Epoch {epoch+1} | D Loss: {loss_D:.4f} | G Loss: {loss_G:.4f}")

print("n")
