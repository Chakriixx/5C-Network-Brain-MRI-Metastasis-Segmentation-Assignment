import torch
import torch.nn as nn

class NestedUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(NestedUNet, self).__init__()
        # Define encoder and decoder layers similar to U-Net, but with dense skip connections
        # Placeholder architecture, customize based on Nested U-Net design
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_enc = self.encoder(x)
        x_out = self.decoder(x_enc)
        return x_out

model_nested_unet = NestedUNet(num_classes=1)
