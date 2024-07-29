import os
import sys
import time
import math
import random
import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import models
import numpy as np
import pdb
import torch.nn.functional as f


resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)


class classifier_generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(classifier_generator, self).__init__()
        self.shared_net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.shared_mu = nn.Linear(hidden_size, output_size)
        self.shared_sigma = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z = self.shared_net(x)
        return self.shared_mu(z), f.softplus(self.shared_sigma(z), beta=1, threshold=20)


class ResNet18_vi(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(ResNet18_vi, self).__init__()
        self.features = nn.Sequential(*list(resnet18.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        self.classifier = classifier_generator(512, 512, 512)

    def forward(self, x):
        x = self.features(x)
        x_f = x.view(x.size(0), -1)
        x = self.fc(x_f)

        return x, x_f

