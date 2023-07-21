""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
from .unet_parts import *

'''
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

'''


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

#'''
#conv19
class conv_block_xx(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_xx, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv01 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch//4, 3, padding=1),
            nn.BatchNorm2d(out_ch//4),
            nn.ReLU(inplace=True)
        )
        self.conv02 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch//4, (1, 3), padding=(0, 2), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch // 4, (7, 1), padding=(6, 0), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True)
        )
        self.conv03 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 4, (3, 1), padding=(2, 0), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//4, out_ch//4, (1, 7), padding=(0,6), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.conv04 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch//4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.conv05 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv05(x)
        x1 = self.conv01(x)
        x2 = self.conv02(x)
        x3 = self.conv03(x)
        x4 = self.conv04(x)
        x_ = torch.cat([x1,x2,x3,x4],dim=1)
        #x111 = self.conv05(x_)
        return x_
'''
class conv_block_xx(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_xx, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.conv01 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//4, 3, padding=1),
            nn.BatchNorm2d(out_ch//4),
            nn.ReLU(inplace=True)
        )
        self.conv02 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//4, (1, 3), padding=(0, 2), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch // 4, (5, 1), padding=(4, 0), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True)
        )
        self.conv03 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, (3, 1), padding=(2, 0), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//4, out_ch//4, (1, 5), padding=(0,4), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.conv04 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.conv05 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv02(x)
        x3 = self.conv03(x)
        x4 = self.conv04(x)
        x_ = torch.cat([x1,x2,x3,x4],dim=1)
        x111 = self.conv05(x_)
        return x111
'''

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self,out_ch=3 ):
        super(UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        in_ch = 3
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_xx(in_ch, filters[0])
        self.Conv2 = conv_block_xx(filters[0], filters[1])
        self.Conv3 = conv_block_xx(filters[1], filters[2])
        self.Conv4 = conv_block_xx(filters[2], filters[3])
        self.Conv5 = conv_block_xx(filters[3], filters[4])

        '''
        self.satten1 = SpatialAttention(16)
        self.satten2 = SpatialAttention(32)
        self.satten3 = SpatialAttention(64)
        self.satten4 = SpatialAttention(128)
        self.satten5 = SpatialAttention(256)
        '''

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block_xx(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block_xx(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block_xx(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block_xx(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        #e11 = self.satten1(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        #e22 = self.satten2(e2)


        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        #e33 = self.satten3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        #e44 = self.satten4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        #e5 = self.satten5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)


        d5 = self.Up_conv5(d5)
        #d5 = self.satten4(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        #d4 = self.satten3(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        #d3 = self.satten2(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #d2 = self.satten1(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)
        outputs = list()
        outputs.append(out)
        return outputs


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