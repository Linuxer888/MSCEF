""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)





class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            # self.conv = DoubleConv(in_channels, out_channels)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            '''
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
            '''

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


###CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=80, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return y * self.sigmoid(out)

class ChannelAttention1(nn.Module):
    def __init__(self, in_planes=80, out_planes=4,ratio=4):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)

        r = max(in_planes//16,4)
        #self.fc = nn.Sequential(nn.Conv2d(in_planes, r, 1, bias=False),
                                #nn.ReLU(),
                                #nn.PReLU(),
                                #nn.Conv2d(r, 4, 1, bias=False))
        self.fc = nn.Conv2d(in_planes, 4, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        '''
        yy = torch.split(x, [16] * 5, dim=1)
        y0 = yy[0]*self.sigmoid(self.fc(self.avg_pool(yy[0])))
        y1 = yy[1] * self.sigmoid(self.fc(self.avg_pool(yy[1])))
        y2 = yy[2] * self.sigmoid(self.fc(self.avg_pool(yy[2])))
        y3 = yy[3] * self.sigmoid(self.fc(self.avg_pool(yy[3])))
        y4 = yy[4] * self.sigmoid(self.fc(self.avg_pool(yy[4])))
        '''
        max_out = self.fc(self.avg_pool(x))
        #out = avg_out + max_out
        ss = self.sigmoid(max_out)
        #print(ss.data)
        yy = torch.split(ss,[1]*4,dim=1)
        #return y * ss
        return yy

class ChannelAttention2(nn.Module):
    def __init__(self, in_planes=80, out_planes=5,ratio=4):
        super(ChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.max_pool = nn.AdaptiveMaxPool2d(1)

        #self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
        #                        nn.ReLU(),
        #                        nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.fc = nn.Conv2d(in_planes, 5, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #y = x
        '''
        yy = torch.split(x, [16] * 5, dim=1)
        y0 = yy[0]*self.sigmoid(self.fc(self.avg_pool(yy[0])))
        y1 = yy[1] * self.sigmoid(self.fc(self.avg_pool(yy[1])))
        y2 = yy[2] * self.sigmoid(self.fc(self.avg_pool(yy[2])))
        y3 = yy[3] * self.sigmoid(self.fc(self.avg_pool(yy[3])))
        y4 = yy[4] * self.sigmoid(self.fc(self.avg_pool(yy[4])))
        '''
        max_out = self.fc(self.avg_pool(x))
        #out = avg_out + max_out
        ss = self.sigmoid(max_out)
        #print(ss.data)
        #print(ss.data)
        yy = torch.split(ss,[1]*5,dim=1)
        xx = torch.split(x,[16]*5,dim=1)
        #return y * ss
        return torch.cat([xx[0]*yy[0],xx[1]*yy[1],xx[2]*yy[2],xx[3]*yy[3],xx[4]*yy[4]],dim=1)

class SELayer(nn.Module):
    def __init__(self, channel=80, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size() # b为batch
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        #return x * y.expand_as(x)
        #print(y.data)
        return torch.mul(x, y.view(b,c,1,1))


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,  k_size=5):##5
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        #print(y.shape) #(4,80,1,1)
        #print(self.conv(y.squeeze(-1).transpose(-1, -2)).shape) #(4,80,1),(4,1,80),(4,1,80)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        #print(y.shape)#(4,80,1,1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        #print(y.expand_as(x).shape)  # (4,80,1,1)


        #print(y.data)
        #y = torch.from_numpy(y_p)
        return x * y.expand_as(x)

class eca_layer_adv(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,  k_size=5):##5
        super(eca_layer_adv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        #print(y.shape) #(4,80,1,1)
        #print(self.conv(y.squeeze(-1).transpose(-1, -2)).shape) #(4,80,1),(4,1,80),(4,1,80)
        # Two different branches of ECA module
        y1 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv(y2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        #print(y.shape)#(4,80,1,1)
        # Multi-scale information fusion
        y_out = self.sigmoid(y1+y2)
        ###y_out = self.sigmoid(y1)
        #print(y.expand_as(x).shape)  # (4,80,1,1)


        #print(y.data)
        #y = torch.from_numpy(y_p)
        return x * y_out.expand_as(x)

class eca_layer_0p(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,  k_size=5):
        super(eca_layer_0p, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2 , bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information

        #y = self.avg_pool(x)
        #print(y.shape) #(4,80,1,1)
        #print(self.conv(y.squeeze(-1).transpose(-1, -2)).shape) #(4,80,1),(4,1,80),(4,1,80)
        # Two different branches of ECA module
        yy = torch.split(x, [16]*5, dim=1)
        y0 = self.conv(self.avg_pool(yy[0]).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y1 = self.conv(self.avg_pool(yy[1]).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv(self.avg_pool(yy[2]).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y3 = self.conv(self.avg_pool(yy[3]).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y4 = self.conv(self.avg_pool(yy[4]).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        #print(y.shape)#(4,80,1,1)
        # Multi-scale information fusion

        r0 = yy[0]*self.sigmoid(y0).expand_as(yy[0])
        r1 = yy[1] * self.sigmoid(y1).expand_as(yy[1])
        r2 = yy[2] * self.sigmoid(y2).expand_as(yy[2])
        r3 = yy[3] * self.sigmoid(y3).expand_as(yy[3])
        r4 = yy[4] * self.sigmoid(y4).expand_as(yy[4])
        #print(y.expand_as(x).shape)  # (4,80,1,1)
        #print(self.sigmoid(y4).expand_as(yy[4]).shape)
        return torch.cat([r0,r1,r2,r3,r4],dim=1)

class eca_layer1(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, inplane1,inplane2, k_size=5):
        super(eca_layer1, self).__init__()
        self.inplane1 = inplane1
        self.inplane2 = inplane2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(inplane1,1,1,bias=False)
        self.fc2 = nn.Conv2d(inplane2,1,1,bias=False)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2 , bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2 , bias=False)
        self.convend = nn.Conv2d(2,1,kernel_size=1,padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        #print(y.shape) #(4,80,1,1)
        #print(self.conv(y.squeeze(-1).transpose(-1, -2)).shape) #(4,80,1),(4,1,80),(4,1,80)
        yy = torch.split(y,[self.inplane1,self.inplane2],dim=1)
        y21 = self.fc1(yy[0]).expand_as(yy[0])
        y22 = self.fc2(yy[1]).expand_as(yy[1])
        y11 = self.conv1(yy[0].squeeze(-1).transpose(-1, -2)).transpose(-1,-2).unsqueeze(-1)
        y12 = self.conv2(yy[1].squeeze(-1).transpose(-1, -2)).transpose(-1,-2).unsqueeze(-1)
        ycat1 = torch.cat([y11,y12],dim=1)
        ycat2 = torch.cat([y21, y22], dim=1)
        ycat = torch.cat([ycat1.transpose(1,2),ycat2.transpose(1,2)],dim=1)
        yend = self.convend(ycat).transpose(1,2)
        # Two different branches of ECA module
        #y11 = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        #y12 = self.conv(yy[1].squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        #print(y.shape)#(4,80,1,1)
        # Multi-scale information fusion

        y = self.sigmoid(yend)
        #print(y.expand_as(x).shape)  # (4,80,1,1)

        return x * y.expand_as(x)

class eca_layer2(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,  k_size=5):
        super(eca_layer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #self.shuffle = channel_shuffle()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=True)
        self.conv2 = nn.Conv1d(1,1, kernel_size=k_size, padding=(k_size)//2,    bias=True)
        #self.conv3 = nn.Conv2d(2,1,kernel_size=1,padding=0,bias=False)
        self.sigmoid = nn.Sigmoid()

    def channel_shuffle(self,x, groups=5):
        batchsize, num_channels, height, width = x.size()

        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)

        # transpose
        # - contiguous() required if transpose() is used before view().
        #   See https://github.com/pytorch/pytorch/issues/764
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x

    def forward(self, x):
        # feature descriptor on the global spatial information
        y1 = self.avg_pool(x)
        y2 = self.channel_shuffle(y1)
        # Two different branches of ECA module
        y1 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv2(y2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.channel_shuffle(y2)
        #y12 = torch.cat([y1.transpose(1,2),y2.transpose(1,2)],dim=1)
        #y12 = self.conv3(y12).transpose(1,2)
        # Multi-scale information fusion
        y12 = y1+y2
        y_end = self.sigmoid(y12)

        return x * y_end.expand_as(x)

class BAM(nn.Module):
    def __init__(self, in_channel=80, reduction_ratio=16, dilation=2):
        super(BAM, self).__init__()
        self.hid_channel = in_channel // reduction_ratio
        self.dilation = dilation
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(in_features=in_channel, out_features=self.hid_channel)
        self.bn1_1d = nn.BatchNorm1d(self.hid_channel)
        self.fc2 = nn.Linear(in_features=self.hid_channel, out_features=in_channel)
        self.bn2_1d = nn.BatchNorm1d(in_channel)

        self.conv1 = nn.Conv2d(in_channel, self.hid_channel,kernel_size=1,padding=0)
        self.bn1_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv2 = nn.Conv2d(self.hid_channel, self.hid_channel, kernel_size=3,stride=1, padding=self.dilation, dilation=self.dilation)
        self.bn2_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv3 = nn.Conv2d(self.hid_channel, self.hid_channel, kernel_size=3,stride=1, padding=self.dilation, dilation=self.dilation)
        self.bn3_2d = nn.BatchNorm2d(self.hid_channel)
        self.conv4 = nn.Conv2d(self.hid_channel,1,kernel_size=1,padding=0)
        self.bn4_2d = nn.BatchNorm2d(1)

    def forward(self, x):
        # Channel attention
        Mc = self.globalAvgPool(x)
        Mc = Mc.view(Mc.size(0), -1)

        Mc = self.fc1(Mc)
        Mc = self.bn1_1d(Mc)
        Mc = self.relu(Mc)

        Mc = self.fc2(Mc)
        Mc = self.bn2_1d(Mc)
        Mc = self.relu(Mc)

        Mc = Mc.view(Mc.size(0), Mc.size(1), 1, 1)

        # Spatial attention
        Ms = self.conv1(x)
        Ms = self.bn1_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv2(Ms)
        Ms = self.bn2_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv3(Ms)
        Ms = self.bn3_2d(Ms)
        Ms = self.relu(Ms)

        Ms = self.conv4(Ms)
        Ms = self.bn4_2d(Ms)
        Ms = self.relu(Ms)

        Ms = Ms.view(x.size(0), 1, x.size(2), x.size(3))
        Mf = 1 + self.sigmoid(Mc * Ms)
        return x * Mf

class MS_CAM(nn.Module):
    '''
    单特征 进行通道加权,作用类似SE模块
    '''

    def __init__(self, channels=80, r=16):#c=64 r=4
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        '''
        self.inplaces = inplanes
        self.conv01 = nn.Conv2d(inplanes, 2, 3, padding=1)
        self.conv02 = nn.Conv2d(2, 2, (1, 3), padding=(0, 2), dilation=2)
        self.conv03 = nn.Conv2d(2, 2, (7, 1), padding=(6, 0), dilation=2)
        self.conv04 = nn.Conv2d(2, 2, (3, 1), padding=(2, 0), dilation=2)
        self.conv05 = nn.Conv2d(2, 2, (1, 7), padding=(0, 6), dilation=2)
        self.conv06 = nn.Conv2d(2, 2, 3, padding=2, dilation=2)
        self.conv07 = nn.Conv2d(2, 2, 3, padding=4, dilation=4)
        # self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(10, 1, kernel_size, padding=0, bias=False)
        '''
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        #self.conv2 = nn.Conv2d(2,1,3,dilation=2,padding=2,bias=True)
        #self.conv12 = nn.Conv2d(2,1,3,padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)


        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #x = torch.cat([avg_out, max_out], dim=1)
        '''
        x1 = self.conv01(x)

        x2 = self.conv02(x1)
        x3 = self.conv03(x2)

        x4 = self.conv04(x1)
        x5 = self.conv05(x4)

        x6 = self.conv06(x1)
        x7 = self.conv07(x6)

        x_ = torch.cat([x1, x3, x5, x6, x7], dim=1)
        '''

        x3 = torch.cat([avg_out,max_out], dim=1)
        x_end1 = self.conv1(x3)

        y = x * self.sigmoid(x_end1)
        return y

class SpatialAttention1(nn.Module):
    def __init__(self, inplane1,inplane2):
        super(SpatialAttention1, self).__init__()
        '''
        self.inplaces = inplanes
        self.conv01 = nn.Conv2d(inplanes, 2, 3, padding=1)
        self.conv02 = nn.Conv2d(2, 2, (1, 3), padding=(0, 2), dilation=2)
        self.conv03 = nn.Conv2d(2, 2, (7, 1), padding=(6, 0), dilation=2)
        self.conv04 = nn.Conv2d(2, 2, (3, 1), padding=(2, 0), dilation=2)
        self.conv05 = nn.Conv2d(2, 2, (1, 7), padding=(0, 6), dilation=2)
        self.conv06 = nn.Conv2d(2, 2, 3, padding=2, dilation=2)
        self.conv07 = nn.Conv2d(2, 2, 3, padding=4, dilation=4)
        # self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(10, 1, kernel_size, padding=0, bias=False)
        '''
        kernel_size = 3
        self.inplace1 = inplane1
        self.inplace2 = inplane2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv2d(2,1,3,dilation=2,padding=2,bias=True)
        self.conv12 = nn.Conv2d(2,1,3,padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        xx = torch.split(x, [self.inplace1,self.inplace2], dim=1)
        # for i in xx:
        y0 = xx[0]
        y1 = xx[1]

        avg_out = torch.mean(y0, dim=1, keepdim=True)
        max_out, _ = torch.max(y0, dim=1, keepdim=True)
        #x = torch.cat([avg_out, max_out], dim=1)
        '''
        x1 = self.conv01(x)

        x2 = self.conv02(x1)
        x3 = self.conv03(x2)

        x4 = self.conv04(x1)
        x5 = self.conv05(x4)

        x6 = self.conv06(x1)
        x7 = self.conv07(x6)

        x_ = torch.cat([x1, x3, x5, x6, x7], dim=1)
        '''

        x3 = torch.cat([avg_out,max_out], dim=1)
        x_end1 = self.conv1(x3)
        x_end2 = self.conv2(x3)
        x33 = torch.cat([x_end1,x_end2],dim=1)
        x_end = self.conv12(x33)
        y2 = y0 * self.sigmoid(x_end)
        y = torch.cat([y2, y1], dim=1)
        return y
class s_c_att(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, inplane1,inplane2 ,k_size=5):
        super(s_c_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv0 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2 , bias=False)
        self.sigmoid = nn.Sigmoid()

        self.c1 = inplane1
        self.c2 = inplane2
        self.conv1 = nn.Conv2d(inplane1, 8, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(inplane2, 8, 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(8, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y0 = self.avg_pool(x)
        #print(y.shape) #(4,80,1,1)
        #print(self.conv(y.squeeze(-1).transpose(-1, -2)).shape) #(4,80,1),(4,1,80),(4,1,80)
        # Two different branches of ECA module
        #y0 = self.sigmoid(self.conv0(y0.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)).expand_as(x)
        y0 = self.conv0(y0.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).expand_as(x)
        #print(y.shape)#(4,80,1,1)
        # Multi-scale information fusion
        #print(y.expand_as(x).shape)  # (4,80,1,1)
        y00 = torch.split(y0,[self.c1, self.c2],dim=1)

        xx = torch.split(x, [self.c1, self.c2], dim=1)
        # for i in xx:
        #y1 = xx[0]
        #y2 = xx[1]

        #avg_out1 = torch.mean(x, dim=1, keepdim=True)
        #max_out1, _ = torch.max(x, dim=1, keepdim=True)
        f1 = self.conv1(xx[0])
        f2 = self.conv2(xx[1])
        f3 = self.relu(f1 + f2)
        #f5 = torch.cat([f3],dim=1)
        f6 = y00[0] + self.conv3(f3)
        y1 = xx[0] * self.sigmoid(f6)
        y2 = xx[1] * self.sigmoid(y00[1])
        y = torch.cat([y1, y2], dim=1)

        return y

#MBGA
class SpatialAttention2(nn.Module):
    def __init__(self, inplane1,inplane2):
        super(SpatialAttention2, self).__init__()
        self.inplace1 = inplane1
        self.inplace2 = inplane2
        self.conv1 = nn.Conv2d(inplane1,8,1,padding=0,bias=True)


        self.conv2 = nn.Conv2d(inplane2,8,1,padding=0,bias=True)

        self.conv3 = nn.Conv2d(8+2,self.inplace1,1,padding=0,bias=True)
        self.conv4 = nn.Conv2d(self.inplace1, self.inplace1//16, 1, groups=self.inplace1//16, bias=True)

        self.conv6 = nn.Conv2d(8+2,1,1,bias=True)
        self.relu = nn.PReLU()#nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xx = torch.split(x, [self.inplace1,self.inplace2], dim=1)
        y1 = xx[0]
        y2 = xx[1]

        avg_out1 = torch.mean(y1, dim=1, keepdim=True)
        max_out1, _ = torch.max(y1, dim=1, keepdim=True)
        f2 = self.conv2(y2)
        f = self.relu(f2)
        f345 = torch.cat([f, avg_out1, max_out1], dim=1)
        f6 = self.conv3(f345)
        y11 = y1 * self.sigmoid(f6)
        y = torch.cat([y11, y2], dim=1)
        return y

class SpatialAttention222(nn.Module):
    def __init__(self, inplane1,inplane2):
        super(SpatialAttention222, self).__init__()
        '''
        self.inplaces = inplanes
        self.conv01 = nn.Conv2d(inplanes, 2, 3, padding=1)
        self.conv02 = nn.Conv2d(2, 2, (1, 3), padding=(0, 2), dilation=2)
        self.conv03 = nn.Conv2d(2, 2, (7, 1), padding=(6, 0), dilation=2)
        self.conv04 = nn.Conv2d(2, 2, (3, 1), padding=(2, 0), dilation=2)
        self.conv05 = nn.Conv2d(2, 2, (1, 7), padding=(0, 6), dilation=2)
        self.conv06 = nn.Conv2d(2, 2, 3, padding=2, dilation=2)
        self.conv07 = nn.Conv2d(2, 2, 3, padding=4, dilation=4)
        # self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(10, 1, kernel_size, padding=0, bias=False)
        '''
        #kernel_size = 3

        self.inplace1 = inplane1
        self.inplace2 = inplane2
        self.num1 = inplane1//16
        self.conv1 = nn.Conv2d(inplane1,8,1,padding=0,bias=True)
        #self.conv11 = nn.Conv2d(inplane1,2*self.num1,1,groups=self.num1)


        self.conv2 = nn.Conv2d(inplane2,8,1,padding=0,bias=True)
        #self.conv3 = nn.Conv2d(,8,1,padding=0,bias=True)
        #self.conv3 = nn.Conv2d(8,1,(1,3),padding=(0,1),bias=True)
        #self.conv4 = nn.Conv2d(8,1,(3,1),padding=(1,0),bias=True)
        #self.conv5 = nn.Conv2d(8,1,1)
        self.conv3 = nn.Conv2d(8+self.num1,self.num1,1,bias=True)
        self.conv4 = nn.Conv2d(inplane1,self.num1,1,groups=self.num1,bias=True)
        self.relu = nn.ReLU(inplace=True)
        #self.conv0 = nn.Conv2d(80,2,1)
        #self.conv123 = nn.Conv2d(3,1,1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        xx = torch.split(x, [self.inplace1,self.inplace2], dim=1)
        # for i in xx:
        y1 = xx[0]
        y2 = xx[1]
        #y3 = xx[2]
        #y4 = xx[3]
        #y5 = xx[4]


        #avg_out1 = torch.mean(x, dim=1, keepdim=True)
        #max_out1, _ = torch.max(x, dim=1, keepdim=True)
        #avg_out2 = torch.mean(y2, dim=1, keepdim=True)
        #max_out2, _ = torch.max(y2, dim=1, keepdim=True)
        #ff = self.conv11(y1)
        f1 = self.conv1(y1)
        f2 = self.conv2(y2)
        #avg_out = torch.mean(x, dim=1, keepdim=True)
        #max_out, _ = torch.max(x, dim=1, keepdim=True)
        #f3 = torch.cat([avg_out, max_out], dim=1)
        f3 = self.relu(f1+f2)
        #f3 = self.conv3(f)
        #f4 = self.conv4(f)
        #f5 = self.conv5(f)
        
        f33 = self.conv4(y1)
        
        #f4 = torch.cat([f3,avg_out1,max_out1],dim=1)
        
        f4 = torch.cat([f3,f33], dim=1)
        f5 = self.sigmoid(self.conv3(f4))
        f5 = torch.split(f5,[1]*self.num1,dim=1)
        #print()
        y11 = torch.split(y1,[16]*self.num1,dim=1)
        #print(y11.shape)
        #uu = f3[0].expand_as(y11[0])
        #y11[0] = y11[0] * (uu)
        temp = y11[0] * f5[0]
        for i in range(1,self.num1):
            temp = torch.cat([temp,y11[i]*f5[i]],dim=1)

        y = torch.cat([temp, y2], dim=1)
        return y
class SpatialAttention22(nn.Module):
    def __init__(self, inplane1,inplane2):
        super(SpatialAttention22, self).__init__()
        '''
        self.inplaces = inplanes
        self.conv01 = nn.Conv2d(inplanes, 2, 3, padding=1)
        self.conv02 = nn.Conv2d(2, 2, (1, 3), padding=(0, 2), dilation=2)
        self.conv03 = nn.Conv2d(2, 2, (7, 1), padding=(6, 0), dilation=2)
        self.conv04 = nn.Conv2d(2, 2, (3, 1), padding=(2, 0), dilation=2)
        self.conv05 = nn.Conv2d(2, 2, (1, 7), padding=(0, 6), dilation=2)
        self.conv06 = nn.Conv2d(2, 2, 3, padding=2, dilation=2)
        self.conv07 = nn.Conv2d(2, 2, 3, padding=4, dilation=4)
        # self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(10, 1, kernel_size, padding=0, bias=False)
        '''
        #kernel_size = 3

        self.inplace1 = inplane1
        self.inplace2 = inplane2
        self.num1 = inplane1//16
        self.conv1 = nn.Conv2d(inplane1,8,1,padding=0,bias=True)


        self.conv2 = nn.Conv2d(inplane2,8,1,padding=0,bias=True)
        #self.conv3 = nn.Conv2d(,8,1,padding=0,bias=True)
        #self.conv3 = nn.Conv2d(8,1,(1,3),padding=(0,1),bias=True)
        #self.conv4 = nn.Conv2d(8,1,(3,1),padding=(1,0),bias=True)
        #self.conv5 = nn.Conv2d(8,1,1)
        self.conv3 = nn.Conv2d(9,1,1,bias=True)
        self.relu = nn.ReLU(inplace=True)
        #self.conv0 = nn.Conv2d(80,2,1)
        #self.conv123 = nn.Conv2d(inplane1,self.num1,1,groups=self.num1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        xx = torch.split(x, [self.inplace1,self.inplace2], dim=1)
        # for i in xx:
        y1 = xx[0]
        y2 = xx[1]
        #y3 = xx[2]
        #y4 = xx[3]
        #y5 = xx[4]


        #avg_out1 = torch.mean(x, dim=1, keepdim=True)
        #max_out1, _ = torch.max(x, dim=1, keepdim=True)
        #avg_out2 = torch.mean(y2, dim=1, keepdim=True)
        #max_out2, _ = torch.max(y2, dim=1, keepdim=True)
        #ff = self.conv0(x)
        f1 = self.conv1(y1)
        f2 = self.conv2(y2)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #max_out, _ = torch.max(x, dim=1, keepdim=True)
        #f3 = torch.cat([avg_out, max_out], dim=1)
        f3 = self.relu(f1+f2)
        #f33 = self.conv123(y1)
        #f4 = self.conv4(f)
        #f5 = self.conv5(f)
        f4 = torch.cat([f3,avg_out],dim=1)
        f5 = self.sigmoid(self.conv3(f4))
        #f5 = torch.split(f5,[1]*self.num1,dim=1)
        #print()
        #y11 = torch.split(y1,[16]*self.num1,dim=1)
        #print(y11.shape)

        #y11[0] = y11[0] * (uu)
        #temp = y11[0] * f5[0]
        #for i in range(1,self.num1):
            #temp = torch.cat([temp,y11[i]*f5[i]],dim=1)

        y = torch.cat([y1*f5, y2], dim=1)
        return y
class SpatialAttention3(nn.Module):
    def __init__(self, inplane1,inplane2):
        super(SpatialAttention3, self).__init__()
        '''
        self.inplaces = inplanes
        self.conv01 = nn.Conv2d(inplanes, 2, 3, padding=1)
        self.conv02 = nn.Conv2d(2, 2, (1, 3), padding=(0, 2), dilation=2)
        self.conv03 = nn.Conv2d(2, 2, (7, 1), padding=(6, 0), dilation=2)
        self.conv04 = nn.Conv2d(2, 2, (3, 1), padding=(2, 0), dilation=2)
        self.conv05 = nn.Conv2d(2, 2, (1, 7), padding=(0, 6), dilation=2)
        self.conv06 = nn.Conv2d(2, 2, 3, padding=2, dilation=2)
        self.conv07 = nn.Conv2d(2, 2, 3, padding=4, dilation=4)
        # self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(10, 1, kernel_size, padding=0, bias=False)
        '''
        #kernel_size = 3

        self.inplace1 = inplane1
        self.inplace2 = inplane2
        self.num1 = inplane1//16
        self.conv1 = nn.Conv2d(inplane1,8,1,padding=0,bias=True)
        #self.conv11 = nn.Conv2d(inplane1,2*self.num1,1,groups=self.num1)


        self.conv2 = nn.Conv2d(inplane2,8,1,padding=0,bias=True)


        for i in range(0,self.num1):
            name = 'conv4_{:d}'.format(i+1)
            setattr(self,name,nn.Conv2d(8+2,1,1,bias=True))
        #self.conv3 = nn.Conv2d(8+2,1,1,bias=True)

        self.conv3 = nn.Conv2d(inplane1,self.num1*2,1,groups=self.num1,bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        xx = torch.split(x, [self.inplace1,self.inplace2], dim=1)
        # for i in xx:
        y1 = xx[0]
        y2 = xx[1]
        #y3 = xx[2]
        #y4 = xx[3]
        #y5 = xx[4]


        #avg_out1 = torch.mean(x, dim=1, keepdim=True)
        #max_out1, _ = torch.max(x, dim=1, keepdim=True)

        f1 = self.conv1(y1)
        f2 = self.conv2(y2)

        #f3 = torch.cat([avg_out, max_out], dim=1)
        f3 = self.relu(f1+f2)

        y11 = torch.split(y1, [16]*self.num1, dim=1)
        f33 = torch.split(self.conv3(y1), [2]*self.num1, dim=1)
        temp = y11[0] * (self.sigmoid(self.conv4_1(torch.cat([f33[0], f3], dim=1))))
        for i in range(1, self.num1):
            key = 'conv4_{:d}'.format(i+1)
            temp = torch.cat([temp, self.sigmoid(y11[i]*(getattr(self, key)(torch.cat([f33[i],f3], dim=1))))], dim=1)
        y = torch.cat([temp, y2], dim=1)

        return y
