import torch
import torch.nn as nn
from torchvision import models


class ResNetBase(nn.Module):
    def __init__(self, avg_output=True, feature_dim=128):
        super().__init__()
        self.avg_output = avg_output
        self.feature_dim = feature_dim
        self.fc = torch.nn.Linear(2048, feature_dim)
        self.logit_dim = 128

        self.num_features = self.feature_dim # for using out class

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNet50(ResNetBase):
    def __init__(self, pretrained='none', feature_dim=2048):
        super(ResNet50, self).__init__(feature_dim=feature_dim)
        if pretrained == 'none':
            print(f'{self.__class__.__name__} no pretrained')
            self.resnet = models.resnet50(weights=None)
        elif pretrained == 'default':
            print(f'{self.__class__.__name__} ImageNet1k pretrained')
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            print(f'{self.__class__.__name__} pretrained: {pretrained}')
            self.resnet = torch.load(pretrained).module

        del self.resnet.fc


class ResNet101(ResNetBase):
    def __init__(self, pretrained='none', feature_dim=2048):
        super(ResNet101, self).__init__(feature_dim=feature_dim)
        if pretrained == 'none':
            print(f'{self.__class__.__name__} no pretrained')
            self.resnet = models.resnet101(weights=None)
        elif pretrained == 'default':
            print(f'{self.__class__.__name__} ImageNet1k pretrained')
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        else:
            print(f'{self.__class__.__name__} pretrained: {pretrained}')
            self.resnet = torch.load(pretrained).module

        del self.resnet.fc


class WideResNet50(ResNetBase):
    def __init__(self, pretrained='none', feature_dim=2048):
        super(WideResNet50, self).__init__(feature_dim=feature_dim)
        if pretrained == 'none':
            print(f'{self.__class__.__name__} no pretrained')
            self.resnet = models.wide_resnet50_2(weights=None)
        elif pretrained == 'default':
            print(f'{self.__class__.__name__} ImageNet1k_v1 pretrained')
            self.resnet = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        else:
            print(f'{self.__class__.__name__} pretrained: {pretrained}')
            self.resnet = torch.load(pretrained).module

        del self.resnet.fc


class WideResNet101(ResNetBase):
    def __init__(self, pretrained='none', feature_dim=2048):
        super(WideResNet101, self).__init__(feature_dim=feature_dim)
        if pretrained == 'none':
            print(f'{self.__class__.__name__} no pretrained')
            self.resnet = models.wide_resnet101_2(weights=None)
        elif pretrained == 'default':
            print(f'{self.__class__.__name__} ImageNet1k_v1 pretrained')
            self.resnet = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.DEFAULT)
        else:
            print(f'{self.__class__.__name__} pretrained: {pretrained}')
            self.resnet = torch.load(pretrained).module

        del self.resnet.fc

