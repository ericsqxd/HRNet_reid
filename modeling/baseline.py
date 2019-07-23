

from torch import nn
import torch
from .cls_hrnet import get_cls_net
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, cfg, num_classes, last_stride, model_path):
        super(Baseline, self).__init__()
        self.stage = get_cls_net(cfg, pretrained=model_path)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.attention_tconv = nn.Conv1d(self.in_planes, 1, 3, padding=1)                        
        self.upsample0 = nn.Sequential(
                                nn.Conv2d(32, self.in_planes, kernel_size=1, stride=1, bias=False),
                                    )
        self.upsample1 = nn.Sequential(
                                nn.Conv2d(64, self.in_planes, kernel_size=1, stride=1, bias=False),
                                    )
        self.upsample2 = nn.Sequential(
                                nn.Conv2d(128, self.in_planes, kernel_size=1, stride=1, bias=False),
                                    )
        self.upsample3 = nn.Sequential(
                                nn.Conv2d(256, self.in_planes, kernel_size=1, stride=1, bias=False),
                                    )
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        y_list = self.stage(x)
        global_feat0 = self.gmp(self.upsample0(y_list[0])) 
        global_feat1 = self.gmp(self.upsample1(y_list[1])) 
        global_feat2 = self.gmp(self.upsample2(y_list[2])) 
        global_feat3 = self.gmp(self.upsample3(y_list[3])) 
        weight_ori = torch.cat([global_feat0, global_feat1, global_feat2, global_feat3], dim=2)
        weight_ori = weight_ori.view(weight_ori.shape[0], weight_ori.shape[1], -1)
        attention_feat = F.relu(self.attention_tconv(weight_ori))
        attention_feat = torch.squeeze(attention_feat)
        weight = F.sigmoid(attention_feat)
        weight = F.normalize(weight, p=1, dim=1)
        #  weight = F.softmax(attention_feat, dim=1)
        #  print(weight.shape)
        weight = torch.unsqueeze(weight, 1)
        weight = weight.expand_as(weight_ori)
        global_feat = torch.mul(weight_ori, weight)
        global_feat = global_feat.sum(-1)
        global_feat = global_feat.view(global_feat.shape[0], -1) #flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            return feat
