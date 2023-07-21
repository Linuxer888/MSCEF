from .backbone_dfa import backbone
from .decoder_dfa import Decoder
import torch
import torch.nn as nn





class DFANet(nn.Module):

    def __init__(self, n_classes=19,  pretrained_backbone=False):
        super(DFANet, self).__init__()
        self.backbone1 = backbone(pretrained=pretrained_backbone)
        self.backbone1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.backbone2 = backbone(pretrained=pretrained_backbone)
        self.backbone2_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.backbone3 = backbone(pretrained=pretrained_backbone)

        self.decoder = Decoder(n_classes=n_classes)


    def forward(self, x):
        enc1_2, enc1_3, enc1_4, fc1, fca1 = self.backbone1(x)
        fca1_up = self.backbone1_up(fca1)

        enc2_2, enc2_3, enc2_4, fc2, fca2 = self.backbone2.forward_concat(fca1_up, enc1_2, enc1_3, enc1_4)
        fca2_up = self.backbone2_up(fca2)

        enc3_2, enc3_3, enc3_4, fc3, fca3 = self.backbone3.forward_concat(fca2_up, enc2_2, enc2_3, enc2_4)

        out = self.decoder(enc1_2, enc2_2, enc3_2, fca1, fca2, fca3)

        return tuple([out])

def get_dfanet(dataset='citys', backbone='', pretrained=False, root='../runs/models',
               pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
        'laser': 'laser'
    }
    from ..data.dataloader import datasets
    model = DFANet(datasets[dataset].NUM_CLASS)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('dfanet_%s_%s_best_model' % (backbone,acronyms[dataset]), root=root),
                              map_location=device))
    return model


def get_dfanet_citys(**kwargs):
    return get_dfanet('citys', **kwargs)

def get_dfanet_laser(**kwargs):
    return get_dfanet('laser', **kwargs)