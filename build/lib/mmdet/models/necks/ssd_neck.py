import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x)))


class ASF(nn.Module):
    # Adaptive Spatial Fusion
    def __init__(self, channel):
        super(ASF, self).__init__()
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(channel*4, int(channel/4), 1),
                                  nn.ReLU(True),
                                  nn.Conv2d(int(channel/4), channel*4, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        out1, out2, out3, out4 = x
        B, C, H, W = out1.size()
        out2_1 = F.interpolate(out2, size=out1.size()[2:], mode='bilinear', align_corners=False)
        out3_1 = F.interpolate(out3, size=out1.size()[2:], mode='bilinear', align_corners=False)
        out4_1 = F.interpolate(out2, size=out1.size()[2:], mode='bilinear', align_corners=False)
        out = torch.cat([out1, out2_1, out3_1, out4_1], 1)
        weight = self.conv(self.GMP(out))
        out = (out * weight).reshape(B, 4, C, H, W).sum(1)
        return out

@NECKS.register_module()
class SSDNECK_ASF(nn.Module):

    def __init__(self):
        super(SSDNECK512, self).__init__()
        
        self.fusion = ASF(512)
        self.conv1 = nn.Conv2d(512, 1024, 1)
        self.conv2 = nn.Conv2d(1024, 512, 1)
        self.conv3 = nn.Conv2d(512, 256, 1)
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.conv5 = nn.Conv2d(256, 256, 1)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        outs = []
        out1, out2, out3, out4, out5, out6 = inputs
        f_out = self.fusion([out1, out2, out3, out4])
        outs.appen(f_out)
        out2m = self.conv1(F.interpolate(f_out, size=out2.size()[2:], mode='bilinear', align_corners=False)) + out2
        outs.append(out2m)
        out3m = self.conv3(F.interpolate(out2m, size=out3.size()[2:], mode='bilinear', align_corners=False)) + out3
        outs.append(out3m)
        out4m = self.conv4(F.interpolate(out3m, size=out4.size()[2:], mode='bilinear', align_corners=False)) + out4
        outs.append(out4m)
        out5m = self.conv5(F.interpolate(out4m, size=out5.size()[2:], mode='bilinear', align_corners=False)) + out5
        outs.append(out5m)
        out6m = self.conv6(F.interpolate(out5m, size=out6.size()[2:], mode='bilinear', align_corners=False)) + out6
        outs.append(out6m)
        return tuple(outs)
    
@NECKS.register_module()
class SSDNECK512(nn.Module):

    def __init__(self):
        super(SSDNECK512, self).__init__()
        
        self.conv1 = nn.Conv2d(1024, 512, 1)
        self.conv2 = nn.Conv2d(512, 1024, 1)
        self.conv3 = nn.Conv2d(256, 512, 1)
        self.conv4 = nn.Conv2d(512, 256, 1)
        self.conv5 = nn.Conv2d(256, 256, 1)
        self.conv6 = nn.Conv2d(256, 256, 1)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        outs = []
        out1, out2, out3, out4, out5, out6, out7 = inputs
        out1m = self.conv1(F.interpolate(out2, size=out1.size()[2:], mode='bilinear', align_corners=False)) + out1
        outs.append(out1m)
        out2m = self.conv2(F.interpolate(out1, size=out2.size()[2:], mode='bilinear', align_corners=False)) + out2
        outs.append(out2m)
        out3m = self.conv3(F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=False)) + out3
        outs.append(out3m)
        out4m = self.conv4(F.interpolate(out3, size=out4.size()[2:], mode='bilinear', align_corners=False)) + out4
        outs.append(out4m)
        out5m = self.conv5(F.interpolate(out6, size=out5.size()[2:], mode='bilinear', align_corners=False)) + out5
        outs.append(out5m)
        out6m = self.conv6(F.interpolate(out5, size=out6.size()[2:], mode='bilinear', align_corners=False)) + out6
        outs.append(out6m)
        outs.append(out7)
        return tuple(outs)
    
@NECKS.register_module()
class SSDNECK(nn.Module):

    def __init__(self):
        super(SSDNECK, self).__init__()
        
        self.conv1 = Conv(1024, 512, 1)
        self.conv2 = Conv(512, 1024, 1)
        self.conv3 = Conv(256, 512, 1)
        self.conv4 = Conv(512, 256, 1)
        self.conv5 = Conv(256, 256, 1)
        self.conv6 = Conv(256, 256, 1)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        outs = []
        out1, out2, out3, out4, out5, out6 = inputs
        out1m = self.conv1(F.interpolate(out2, size=out1.size()[2:], mode='bilinear', align_corners=False)) + out1
        outs.append(out1m)
        out2m = self.conv2(F.interpolate(out1, size=out2.size()[2:], mode='bilinear', align_corners=False)) + out2
        outs.append(out2m)
        out3m = self.conv3(F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=False)) + out3
        outs.append(out3m)
        out4m = self.conv4(F.interpolate(out3, size=out4.size()[2:], mode='bilinear', align_corners=False)) + out4
        outs.append(out4m)
        out5m = self.conv5(F.interpolate(out6, size=out5.size()[2:], mode='bilinear', align_corners=False)) + out5
        outs.append(out5m)
        out6m = self.conv6(F.interpolate(out5, size=out6.size()[2:], mode='bilinear', align_corners=False)) + out6
        outs.append(out6m)
        
        return tuple(outs)

# @NECKS.register_module()
# class SSDNECK(nn.Module):

#     def __init__(self):
#         super(SSDNECK, self).__init__()
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()
        
#         self.conv1 = nn.Conv2d(1024, 512, 1)
#         self.conv2 = nn.Conv2d(512, 1024, 1)
#         self.conv3 = nn.Conv2d(256, 512, 1)
#         self.conv4 = nn.Conv2d(512, 256, 1)
#         self.conv5 = nn.Conv2d(256, 256, 1)
#         self.conv6 = nn.Conv2d(256, 256, 1)

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         """Initialize the weights of FPN module."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform', bias=0)

#     @auto_fp16()
#     def forward(self, inputs):
#         """Forward function."""
#         outs = []
#         out1, out2, out3, out4, out5, out6 = inputs
#         out1m = self.conv1(F.interpolate(out2, size=out1.size()[2:], mode='bilinear', align_corners=False)) + out1
#         outs.append(out1m)
#         out2m = self.conv2(F.interpolate(out1, size=out2.size()[2:], mode='bilinear', align_corners=False)) + out2
#         outs.append(out2m)
#         out3m = self.conv3(F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=False)) + out3
#         outs.append(out3m)
#         out4m = self.conv4(F.interpolate(out3, size=out4.size()[2:], mode='bilinear', align_corners=False)) + out4
#         outs.append(out4m)
#         out5m = self.conv5(F.interpolate(out6, size=out5.size()[2:], mode='bilinear', align_corners=False)) + out5
#         outs.append(out5m)
#         out6m = self.conv6(F.interpolate(out5, size=out6.size()[2:], mode='bilinear', align_corners=False)) + out6
#         outs.append(out6m)
        
#         return tuple(outs)


# @NECKS.register_module()
# class SSDNECK(nn.Module):

#     def __init__(self):
#         super(SSDNECK, self).__init__()
        
#         self.conv1 = nn.Conv2d(512, 1024, 1)
#         self.conv2 = nn.Conv2d(1024, 512, 1)
#         self.conv3 = nn.Conv2d(512, 1024, 1)
#         self.conv4 = nn.Conv2d(1024, 512, 1)
#         self.conv5 = nn.Conv2d(256, 256, 1)
#         self.conv6 = nn.Conv2d(256, 256, 1)
#         self.conv7 = nn.Conv2d(256, 256, 1)
#         self.conv8 = nn.Conv2d(256, 256, 1)

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         """Initialize the weights of FPN module."""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform', bias=0)

#     @auto_fp16()
#     def forward(self, inputs):
#         """Forward function."""
#         outs = []
#         out1, out2, out3, out4, out5, out6 = inputs
#         out2_3 = self.conv1(F.interpolate(out3, size=out2.size()[2:], mode='bilinear', align_corners=False)) + out2
#         out2_3_1 = self.conv2(F.interpolate(out2_3, size=out1.size()[2:], mode='bilinear', align_corners=False)) + out1
#         outs.append(out2_3_1)
#         out2_1 = self.conv3(F.interpolate(out1, size=out2.size()[2:], mode='bilinear', align_corners=False)) + out2
#         out2_1_3 = self.conv4(F.interpolate(out2_1, size=out3.size()[2:], mode='bilinear', align_corners=False)) + out3
#         outs.append(out2_3 + out2_1)
#         outs.append(out2_1_3)
        
#         out5_6 = self.conv5(F.interpolate(out6, size=out5.size()[2:], mode='bilinear', align_corners=False)) + out5
#         out5_6_4 = self.conv6(F.interpolate(out5_6, size=out4.size()[2:], mode='bilinear', align_corners=False)) + out4
#         outs.append(out5_6_4)
#         out5_4 = self.conv7(F.interpolate(out4, size=out5.size()[2:], mode='bilinear', align_corners=False)) + out5
#         out5_4_6 = self.conv8(F.interpolate(out5_4, size=out6.size()[2:], mode='bilinear', align_corners=False)) + out6
#         outs.append(out5_6 + out5_4)
#         outs.append(out5_4_6)
        
#         return tuple(outs)