# import torch
# import torch.nn as nn
# import torchvision
# import torch.nn.functional as F

# class MobileNetUNet(nn.Module):

#     def __init__(self, num_classes=21):

#         super().__init__()

#         backbone = torchvision.models.mobilenet_v2(pretrained=True).features

#         self.enc1 = backbone[:2]
#         self.enc2 = backbone[2:4]
#         self.enc3 = backbone[4:7]
#         self.enc4 = backbone[7:14]
#         self.enc5 = backbone[14:]

#         self.up4 = nn.ConvTranspose2d(1280,320,2,stride=2)
#         self.conv4 = nn.Conv2d(320+96,256,3,padding=1)

#         self.up3 = nn.ConvTranspose2d(256,96,2,stride=2)
#         self.conv3 = nn.Conv2d(96+32,128,3,padding=1)

#         self.up2 = nn.ConvTranspose2d(128,32,2,stride=2)
#         self.conv2 = nn.Conv2d(32+24,64,3,padding=1)

#         self.up1 = nn.ConvTranspose2d(64,24,2,stride=2)
#         self.conv1 = nn.Conv2d(24+16,32,3,padding=1)

#         self.final = nn.Conv2d(32,num_classes,1)

#     def forward(self,x):

#         e1 = self.enc1(x)
#         e2 = self.enc2(e1)
#         e3 = self.enc3(e2)
#         e4 = self.enc4(e3)
#         e5 = self.enc5(e4)

#         d4 = self.up4(e5)
#         e4 = F.interpolate(e4,size=d4.shape[2:],mode="bilinear",align_corners=False)
#         d4 = torch.cat([d4,e4],dim=1)
#         d4 = self.conv4(d4)

#         d3 = self.up3(d4)
#         e3 = F.interpolate(e3,size=d3.shape[2:],mode="bilinear",align_corners=False)
#         d3 = torch.cat([d3,e3],dim=1)
#         d3 = self.conv3(d3)

#         d2 = self.up2(d3)
#         e2 = F.interpolate(e2,size=d2.shape[2:],mode="bilinear",align_corners=False)
#         d2 = torch.cat([d2,e2],dim=1)
#         d2 = self.conv2(d2)

#         d1 = self.up1(d2)
#         e1 = F.interpolate(e1,size=d1.shape[2:],mode="bilinear",align_corners=False)
#         d1 = torch.cat([d1,e1],dim=1)
#         d1 = self.conv1(d1)

#         out = self.final(d1)

#         out = F.interpolate(out,size=(300,300),mode="bilinear",align_corners=False)

#         return out

# import torch
# import torch.nn as nn
# import torchvision
# from torchvision.models import MobileNet_V3_Large_Weights

# class SegFormerLite(nn.Module):

#     def __init__(self, num_classes=21):
#         super().__init__()

#         backbone = torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).features

#         self.stage1 = backbone[:4]
#         self.stage2 = backbone[4:7]
#         self.stage3 = backbone[7:13]
#         self.stage4 = backbone[13:]

#         self.conv1 = nn.Conv2d(24,128,1)
#         self.conv2 = nn.Conv2d(40,128,1)
#         self.conv3 = nn.Conv2d(112,128,1)
#         self.conv4 = nn.Conv2d(960,128,1)

#         self.fuse = nn.Conv2d(512,256,3,padding=1)

#         self.head = nn.Sequential(
#             nn.Conv2d(256,128,3,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128,num_classes,1)
#         )

#     def forward(self,x):

#         s1 = self.stage1(x)
#         s2 = self.stage2(s1)
#         s3 = self.stage3(s2)
#         s4 = self.stage4(s3)

#         s1 = self.conv1(s1)
#         s2 = self.conv2(s2)
#         s3 = self.conv3(s3)
#         s4 = self.conv4(s4)

#         size = s1.shape[2:]

#         s2 = nn.functional.interpolate(s2,size=size,mode="bilinear",align_corners=False)
#         s3 = nn.functional.interpolate(s3,size=size,mode="bilinear",align_corners=False)
#         s4 = nn.functional.interpolate(s4,size=size,mode="bilinear",align_corners=False)

#         x = torch.cat([s1,s2,s3,s4],dim=1)

#         x = self.fuse(x)

#         out = self.head(x)

#         out = nn.functional.interpolate(out,size=(300,300),mode="bilinear",align_corners=False)

#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x0_5


class LightSegNet(nn.Module):

    def __init__(self, num_classes=21):
        super().__init__()

        backbone = shufflenet_v2_x0_5(pretrained=True)

        self.stage1 = nn.Sequential(
            backbone.conv1,
            backbone.maxpool
        )

        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4

        # channel reduction
        self.conv1 = nn.Conv2d(24,16,1)
        self.conv2 = nn.Conv2d(48,16,1)
        self.conv3 = nn.Conv2d(96,16,1)
        self.conv4 = nn.Conv2d(192,16,1)

        # depthwise fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,groups=64),
            nn.Conv2d(64,32,1),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Conv2d(32,16,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,num_classes,1)
        )

    def forward(self,x):

        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        s1 = self.conv1(s1)
        s2 = self.conv2(s2)
        s3 = self.conv3(s3)
        s4 = self.conv4(s4)

        size = s1.shape[2:]

        s2 = F.interpolate(s2,size=size,mode="bilinear",align_corners=False)
        s3 = F.interpolate(s3,size=size,mode="bilinear",align_corners=False)
        s4 = F.interpolate(s4,size=size,mode="bilinear",align_corners=False)

        x = torch.cat([s1,s2,s3,s4],dim=1)

        x = self.fuse(x)

        out = self.head(x)

        out = F.interpolate(out,size=(300,300),mode="bilinear",align_corners=False)

        return out