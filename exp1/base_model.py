import torch
import torch.nn as nn
import torchvision


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.features = vgg16.features[:23]
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs
