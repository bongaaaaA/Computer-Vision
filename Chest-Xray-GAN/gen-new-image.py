import torch
from torchvision.utils import save_image
from models.conditional_generator import ConditionalGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = ConditionalGenerator().to(device)
G.load_state_dict(torch.load("models/conditional_generator.pth"))

labels = torch.tensor([0, 1]*8).to(device)  # generate 16 images
noise = torch.randn(16, 100, 1, 1).to(device)
fake_imgs = G(noise, labels)
save_image(fake_imgs, "generated_xray.png", normalize=True, nrow=4)

#done gen new images