import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import VGG, constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import ResNet

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

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

@BACKBONES.register_module()
class SSDRES(nn.Module):
    def __init__(self,
                 input_size,
                 depth=50,
                 pretrained=None):
        super(SSDRES, self).__init__()
        assert input_size in (300, 512)
        self.input_size = input_size
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        backbone = ResNet(depth=depth)
        if pretrained is not None:
            print("load pretrained weights from {}".format(pretrained))
            model_dict = torch.load(pretrained)['state_dict']
            state_dict = backbone.state_dict()
            old_dict = {}
            for k,v in state_dict.items():
                if ('backbone.' + k) in model_dict.keys() and v.size() == model_dict['backbone.' + k].size():
                    old_dict[k] = model_dict['backbone.' + k]
            missing_keys, unexpected_keys = backbone.load_state_dict(old_dict)
            print("missing kyes:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        # 修改conv4_block1的步距，从2->1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)
        self._build_additional_features(self.out_channels)
    
    def _build_additional_features(self, input_size):
        """
        为backbone(resnet50)添加额外的一系列卷积层，得到相应的一系列特征提取器
        :param input_size:
        :return:
        """
        additional_blocks = []
        # input_size = [1024, 512, 512, 256, 256, 256] for resnet50
        middle_channels = [256, 256, 128, 128, 128]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        self.extra = nn.ModuleList(additional_blocks)
    
    def forward(self, x):
        """Forward function."""
        outs = []
        x = self.feature_extractor(x) # [N, 1024, 38, 38]
        outs.append(x)
        for layer in self.extra:
            x = layer(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)
        
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
#         if isinstance(pretrained, str):
#             torch.load_state_dict(torch.load(pretrained), strict=False)
#         elif pretrained is None:
#             for m in self.features.modules():
#                 if isinstance(m, nn.Conv2d):
#                     kaiming_init(m)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     constant_init(m, 1)
#                 elif isinstance(m, nn.Linear):
#                     normal_init(m, std=0.01)
#         else:
#             raise TypeError('pretrained must be a str or None')
            
        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')