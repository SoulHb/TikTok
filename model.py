import segmentation_models_pytorch as smp
import torch
from torch import nn


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.unet = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
        for name, param in self.named_parameters():
            if name == 'unet.encoder.layer4.2.bn2.bias':
                break
            param.requires_grad = False

    def forward(self, x):
        return self.unet(x)