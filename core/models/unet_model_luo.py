""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
from .unet_parts import *
from .init_weights import init_weights


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


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

#'''
#conv19
class conv_block_xx(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_xx, self).__init__()

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

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
                # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
        '''
    def forward(self, x):
        x = self.conv05(x)

        x1 = self.conv01(x)

        x2 = self.conv02(x)

        x3 = self.conv03(x)

        x4 = self.conv04(x)

        x_ = torch.cat([x1,x2,x3,x4],dim=1)

        return x_


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
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
        '''
    def forward(self, x):
        x = self.up(x)
        return x
class up_conv11(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv11, self).__init__()

        self.up = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
        '''
    def forward(self, x):

        x = self.up(x)
        return x
class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, out_ch=2 ):
        super(UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        in_ch = 3
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        '''
        #self.shuffle = ShuffleBlock(4)
        self.eca2 = eca_layer()
        self.eca3 = eca_layer()
        self.eca4 = eca_layer()
        self.eca5 = eca_layer()
        self.eca6 = eca_layer()
        self.eca7 = eca_layer()
        self.eca8 = eca_layer()
        '''

        self.Up5 = up_conv11(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv11(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
        '''
    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)


        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)


        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        #print(e4.shape)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = F.interpolate(e5,(37,37))
        d5 = self.Up5(d5)
        #print(d5.shape)
        d5 = torch.cat((e4, d5), dim=1)


        d5 = self.Up_conv5(d5)

        d5 = F.interpolate(d5,(75,75))
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

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