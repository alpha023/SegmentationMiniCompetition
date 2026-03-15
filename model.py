import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x0_5

class DSConv(nn.Sequential):
    """Depthwise Separable Convolution: Very low FLOPs, high efficiency."""
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__(
            # Depthwise: groups=in_chan ensures each channel is filtered separately
            nn.Conv2d(in_chan, in_chan, 3, stride, 1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            # Pointwise: 1x1 conv to change channel depth
            nn.Conv2d(in_chan, out_chan, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

class LightweightUNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Load backbone
        backbone = shufflenet_v2_x0_5(weights='DEFAULT')

        # Encoder stages
        self.init_conv = nn.Sequential(backbone.conv1, backbone.maxpool) # 24ch
        self.stage2 = backbone.stage2 # 48ch
        self.stage3 = backbone.stage3 # 96ch
        self.stage4 = backbone.stage4 # 192ch

        # Global Context Branch (The "Dice Booster")
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(192, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Lateral 1x1 convs to unify scales to 64 channels each
        self.lat4 = nn.Conv2d(192, 64, 1)
        self.lat3 = nn.Conv2d(96, 64, 1)
        self.lat2 = nn.Conv2d(48, 64, 1)

        # Refinement: Input is 256 (64*4 branches), Output reduced to 64
        self.refine = DSConv(256, 64) 
        
        self.head = nn.Sequential(
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        
        # --- Encoder ---
        s1 = self.init_conv(x) # 1/4
        s2 = self.stage2(s1)   # 1/8
        s3 = self.stage3(s2)   # 1/16
        s4 = self.stage4(s3)   # 1/32

        # --- Decoder Branching ---
        # 1. Global Context
        g = self.gap(s4)
        g = F.interpolate(g, size=s2.shape[2:], mode='bilinear', align_corners=False)

        # 2. Top-down paths (all upsampled to s2/stage2 resolution)
        p4 = F.interpolate(self.lat4(s4), size=s2.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.lat3(s3), size=s2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lat2(s2)

        # --- Fusion ---
        # Concatenate 4 branches: 64 + 64 + 64 + 64 = 256 channels
        out = torch.cat([p2, p3, p4, g], dim=1) 
        
        # Refine combined features
        out = self.refine(out) 
        out = self.head(out)
        
        # Final Upsample to original image resolution
        return F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x0_5

class DSConv(nn.Sequential):
    """Depthwise Separable Convolution: Very low FLOPs, high efficiency."""
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__(
            # Depthwise: groups=in_chan ensures each channel is filtered separately
            nn.Conv2d(in_chan, in_chan, 3, stride, 1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            # Pointwise: 1x1 conv to change channel depth
            nn.Conv2d(in_chan, out_chan, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

class LightweightUNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Load backbone
        backbone = shufflenet_v2_x0_5(weights='DEFAULT')

        # Encoder stages
        self.init_conv = nn.Sequential(backbone.conv1, backbone.maxpool) # 24ch
        self.stage2 = backbone.stage2 # 48ch
        self.stage3 = backbone.stage3 # 96ch
        self.stage4 = backbone.stage4 # 192ch

        # Global Context Branch (The "Dice Booster")
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(192, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Lateral 1x1 convs to unify scales to 64 channels each
        self.lat4 = nn.Conv2d(192, 48, 1)
        self.lat3 = nn.Conv2d(96, 48, 1)
        self.lat2 = nn.Conv2d(48, 48, 1)

        # Refinement: Input is 256 (64*4 branches), Output reduced to 64
        self.refine = DSConv(192, 48) 
        
        self.head = nn.Sequential(
            nn.Conv2d(48, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        
        # --- Encoder ---
        s1 = self.init_conv(x) # 1/4
        s2 = self.stage2(s1)   # 1/8
        s3 = self.stage3(s2)   # 1/16
        s4 = self.stage4(s3)   # 1/32

        # --- Decoder Branching ---
        # 1. Global Context
        g = self.gap(s4)
        g = F.interpolate(g, size=s2.shape[2:], mode='bilinear', align_corners=False)

        # 2. Top-down paths (all upsampled to s2/stage2 resolution)
        p4 = F.interpolate(self.lat4(s4), size=s2.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.lat3(s3), size=s2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lat2(s2)

        # --- Fusion ---
        # Concatenate 4 branches: 64 + 64 + 64 + 64 = 256 channels
        out = torch.cat([p2, p3, p4, g], dim=1) 
        
        # Refine combined features
        out = self.refine(out) 
        out = self.head(out)
        
        # Final Upsample to original image resolution
        return F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

