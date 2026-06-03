import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # --- Encoder ---
        # Block 1: 40x50 -> 20x25
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)

        # Block 2: 20x25 -> 10x12
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)

        # Block 3: 10x12 -> 5x6
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # --- Decoder ---
        # Up 1: 5x6 -> 10x12
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 256 input because of concat
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Up 2: 10x12 -> 20x25
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Up 3: 20x25 -> 40x50
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Final output layer
            nn.Conv2d(32, 1, kernel_size=1) 
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with Force Alignment
        # We resize 'd1' to match 'e3' exactly because 25/2 is odd and causes shape mismatch
        d1 = self.up1(b)
        if d1.shape != e3.shape: d1 = F.interpolate(d1, size=e3.shape[2:])
        d1 = torch.cat((d1, e3), dim=1) # Skip Connection
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        if d2.shape != e2.shape: d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat((d2, e2), dim=1) # Skip Connection
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        if d3.shape != e1.shape: d3 = F.interpolate(d3, size=e1.shape[2:])
        d3 = torch.cat((d3, e1), dim=1) # Skip Connection
        #out = self.dec3(d3)     #orignal has not it. 
        delta = self.dec3(d3)          # raw output, Predict the correction (can be positive or negative) #original has this
        out = torch.clamp(x + delta, 0, 1)  #original has this
        
        return out     #original has this
        #return torch.sigmoid(out)