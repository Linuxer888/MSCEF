""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_classes=2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 16)
        #self.catt1 = ChannelAttention(64)
        self.satten1 = SpatialAttention(16)
        self.down1 = Down(16, 32)
        #self.catt2 = ChannelAttention(128)
        self.satten2 = SpatialAttention(32)
        self.down2 = Down(32, 64)
        #self.catt3 = ChannelAttention(256)
        self.satten3 = SpatialAttention(64)
        self.down3 = Down(64, 128) #256,512
        #self.catt4 = ChannelAttention(256)
        self.satten4 = SpatialAttention(128)
        factor = 2 if bilinear else 1
        #factor = 1

        self.down4 = Down(128, 256 // factor)
        self.satten5 = SpatialAttention(256//factor)
        self.up1 = Up(256, 128 // factor, bilinear)

        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)



        self.__setattr__('exclusive', ['inc', 'down1', 'down2', 'down3','down4',
                                       'satten1','satten2','satten3','satten4',
                                       'satten5',
                                       'up1','up2', 'up3', 'up4', 'outc'])

    def forward(self, x):
        x1 = self.inc(x)
        #x1 = self.catt1(x1)
        x1 = self.satten1(x1)
        x2 = self.down1(x1)
        #x2 = self.catt2(x2)
        x2 = self.satten2(x2)
        x3 = self.down2(x2)
        #x3 = self.catt3(x3)
        x3 = self.satten3(x3)
        x4 = self.down3(x3)
        #x4 = self.catt4(x4)
        x4 = self.satten4(x4)

        x5 = self.down4(x4)
        x5 = self.satten5(x5)
        x = self.up1(x5, x4)

        x = self.up2(x, x3) #x,x3
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        outputs = list()
        outputs.append(logits)
        return tuple(outputs)


def get_unet(dataset='pascal_voc', backbone='', pretrained=False, root='../runs/models',
               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
        'laser': 'laser'
    }
    from core.data.dataloader import datasets
    model = UNet(datasets[dataset].NUM_CLASS)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('unet_%s_%s_best_model' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model