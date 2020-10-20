import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, models
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer

class ConvPosenet(BaseNetwork):
    def __init__(self, opt):
        super().__init__()

        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, 2048)
        self.fc = nn.Linear(2048, 3)

        self.init_modules = [self.feature_extractor.fc, self.fc]

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.feature_extractor(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)
        pose = self.fc(x)

        return pose

    def init_weights(self, init_type='normal', gain=0.02):
        for m in self.init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
