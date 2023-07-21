# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import unetConv2, unetUp_origin,unetUp_origin_xx,unetConv2_xx
from .init_weights import init_weights
import numpy as np
from torchvision import models


class UNet_2Plus(nn.Module):

    def __init__(self,  n_classes=1, in_channels=3,  is_ds=False):
        super(UNet_2Plus, self).__init__()

        self.in_channels = in_channels

        self.is_ds = is_ds

        #filters = [32, 64, 128, 256, 512]
        filters = [16, 32, 64, 128, 256]
        #filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0])
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4])


        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0])
        self.up_concat11 = unetUp_origin(filters[2], filters[1])
        self.up_concat21 = unetUp_origin(filters[3], filters[2])
        self.up_concat31 = unetUp_origin(filters[4], filters[3])#xx

        self.up_concat02 = unetUp_origin(filters[1], filters[0], 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], 3)#xx

        self.up_concat03 = unetUp_origin(filters[1], filters[0], 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], 4)#xx

        self.up_concat04 = unetUp_origin(filters[1], filters[0], 5)#xx

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        '''
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
        '''
        '''
        self.__setattr__('exclusive', ['conv00','conv10','conv20','conv30','conv40',
                                       'maxpool0', 'maxpool1','maxpool2','maxpool3',
                                       'up_concat01', 'up_concat11', 'up_concat21', 'up_concat31',
                                       'up_concat02','up_concat12','up_concat22',
                                       'up_concat03', 'up_concat13','up_concat04',
                                       'final_1','final_2','final_3','final_4'])
        '''

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)
        maxpool0 = self.maxpool0(X_00)
        X_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(X_10)
        X_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(X_20)
        X_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(X_30)
        X_40 = self.conv40(maxpool3)

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4
        outputs = list()
        if self.is_ds:
            outputs.append(final)
            return tuple(outputs)
        else:
            outputs.append(final_4)
            return tuple(outputs)
'''
model = UNet_2Plus()
print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters())/1000000)
params = list(model.named_parameters())
for i in range(len(params)):
    (name, param) = params[i]
    print(name)
    print(param.shape)
'''
def get_unet2plus(dataset='pascal_voc', backbone='', pretrained=False, root='../runs/models',
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
    model = UNet_2Plus(datasets[dataset].NUM_CLASS)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('unet2plus_%s_%s_best_model' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model