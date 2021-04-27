# author: Nikola Zubic

import torch.nn as nn
import torchvision.models as models
import torch


class ResNet(nn.Module):
    """
    The main problem with this dataset is the usage of 4-channel input (RGBY) which limits the usage of pretrained
    ImageNet models. Thus, we replace the first convolutional layer from 7x7x3 -> 64 to 7x7x4 -> 64 while keeping the
    weights from 3 -> 64. The weights for the last channel can be initialized with values from other channels (we will
    retrain it).
    """
    def __init__(self, pretrained=True):
        super().__init__()

        encoder = models.resnet34(pretrained=pretrained)

        self.convolutional_layer_1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, stride=2, padding=3)

        if pretrained:
            w = encoder.convolutional_layer_1.weight

            self.convolutional_layer_1.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])),
                                                                       dim=1))

        self.bn1 = encoder.bn1
        self.ReLU = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = nn.Sequential(self.convolutional_layer_1, self.relu, self.bn1, self.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4  # the head will be added automatically

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
