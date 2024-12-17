# This class was taken directly from: https://github.com/hfslyc/AdvSemiSeg/blob/master/model/discriminator.py
# Editted with chatgpt
import torch.nn as nn
import torch.nn.functional as F

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()
        self.relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        # Encoder (downsampling)
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)  # Output: (ndf, H/2, W/2)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)        # Output: (ndf*2, H/4, W/4)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)      # Output: (ndf*4, H/8, W/8)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)      # Output: (ndf*8, H/16, W/16)

        # Decoder (upsampling)
        self.deconv1 = nn.ConvTranspose2d(ndf*8, ndf*4, kernel_size=4, stride=2, padding=1)  # Output: (ndf*4, H/8, W/8)
        self.deconv2 = nn.ConvTranspose2d(ndf*4, ndf*2, kernel_size=4, stride=2, padding=1)  # Output: (ndf*2, H/4, W/4)
        self.deconv3 = nn.ConvTranspose2d(ndf*2, ndf, kernel_size=4, stride=2, padding=1)    # Output: (ndf, H/2, W/2)
        self.deconv4 = nn.ConvTranspose2d(ndf, num_classes, kernel_size=4, stride=2, padding=1)  # Output: (num_classes, H, W)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.Sigmoid(x)
        return x