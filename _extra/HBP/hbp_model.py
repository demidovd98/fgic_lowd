import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .resnet_model import resnet50


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        ## Modify dimensions based on model

        # resnet 18,34
        # self.proj0 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)
        # self.proj1 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)
        # self.proj2 = nn.Conv2d(512, 8192, kernel_size=1, stride=1)

        # resnet 50, ...
        self.proj0 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)
        self.proj1 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(2048, 8192, kernel_size=1, stride=1)

        self.avgpool = nn.AvgPool2d(kernel_size=7) #changed from 14

        self.features = resnet50(pretrained=True,
                                              model_root='/home/mihail.mihaylov/Desktop/phd/fgic_lowd/_extra/HBP/resnet50-19c8e357.pth')

        self.fc_concat =  torch.nn.Linear(8192 * 3, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):

        batch_size = x.size(0)
        feature4_0, feature4_1, feature4_2 = self.features(x)


        feature4_0 = self.proj0(feature4_0)
        feature4_1 = self.proj1(feature4_1)
        feature4_2 = self.proj2(feature4_2)

        inter1 = feature4_0 * feature4_1
        inter2 = feature4_0 * feature4_2
        inter3 = feature4_1 * feature4_2

        inter1 = self.avgpool(inter1).view(batch_size, -1)
        inter2 = self.avgpool(inter2).view(batch_size, -1)
        inter3 = self.avgpool(inter3).view(batch_size, -1)


        result1 = torch.nn.functional.normalize(torch.sign(inter1) * torch.sqrt(torch.abs(inter1) + 1e-10))
        result2 = torch.nn.functional.normalize(torch.sign(inter2) * torch.sqrt(torch.abs(inter2) + 1e-10))
        result3 = torch.nn.functional.normalize(torch.sign(inter3) * torch.sqrt(torch.abs(inter3) + 1e-10))


        features = torch.cat((result1, result2, result3), 1)
        result = self.fc_concat(features) 
        return features, result
        return result
    
        features = torch.cat((result1, result2, result3), 1)
        logits = self.fc_concat(features) 
        logits= self.softmax(logits)
        return features, logits
