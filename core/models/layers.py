import torch
import torch.nn as nn

import torch.nn.functional as F
from .init_weights import init_weights
from .unet_parts import *
'''
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        
        for i in range(1, n + 1):
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                 nn.BatchNorm2d(out_size),
                                 nn.ReLU(inplace=True), )
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size
        
        #‘’‘’‘’zhushile 
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
        
    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x
'''
class unetConv2(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(unetConv2, self).__init__()

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

class unetConv2_xx(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(unetConv2_xx, self).__init__()

        self.conv01 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 4, 3, padding=1),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True)
        )

        self.conv04 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.conv02 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 4, (1, 3), padding=(0, 2), dilation=2),
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
            nn.Conv2d(out_ch // 4, out_ch // 4, (1, 7), padding=(0, 6), dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
        )

        self.att = ChannelAttention1(out_ch)
        '''
        self.conv01 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 4, 3, padding=1),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True)
        )

        self.conv04 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch//4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True),
        )
        self.conv02 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch//2, (1, 3), padding=(0, 2), dilation=2),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch // 2, (7, 1), padding=(6, 0), dilation=2),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True)
        )

        self.conv03 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // 2, (3, 1), padding=(2, 0), dilation=2),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//2, out_ch//2, (1, 7), padding=(0,6), dilation=2),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
        )
        '''

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
        att = self.att(x_)
        y = torch.cat([x1 * att[0], x2 * att[1], x3 * att[2], x4 * att[3]], dim=1)
        return y

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size*2, out_size)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        '''
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')
        '''
    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
    
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)

        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size)
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_size,out_size,3,padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

        '''
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')
        '''
    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)

class unetUp_origin_xx(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2):
        super(unetUp_origin_xx, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)

        self.conv = unetConv2_xx(in_size + (n_concat - 2) * out_size, out_size)
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_size,out_size,3,padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )

        '''
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')
        '''
    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
