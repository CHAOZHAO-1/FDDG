from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Distribution Generator
class GeneDistrNet(nn.Module):
    def __init__(self,input_size,hidden_size, optimizer,lr,weight_decay):
        super(GeneDistrNet,self).__init__()
        self.num_labels = 3
        self.latent_size = 256
        self.genedistri = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(input_size + self.num_labels, self.latent_size)),###为什么+num_labels?
            ("relu1", nn.LeakyReLU()),

            ("fc2", nn.Linear(self.latent_size, hidden_size)),
            ("relu2", nn.ReLU()),
        ]))
        self.optimizer = optimizer(self.genedistri.parameters(), lr=lr, weight_decay=weight_decay)
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.5)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = x.cuda()
        x = self.genedistri(x)
        return x

# Feature Extractor
class feature_extractor(nn.Module):
    # Reference from https://github.com/belaalb/G2DM/blob/master/vlcs-ours/models.py
    def __init__(self, optimizer, lr, weight_decay, num_classes=7):
        super(feature_extractor,self).__init__()
        self.num_classes = num_classes



        self.features = nn.Sequential(OrderedDict([

            ("conv1",nn.Conv1d(1, 16, kernel_size=64, stride=1)),  # 32, 24, 24
            ("bano1",nn.BatchNorm1d(16)),
            ("relu1",nn.ReLU(inplace=True)),
            ("pool1",nn.MaxPool1d(kernel_size=2, stride=2)),

            ("conv2", nn.Conv1d(16, 32, kernel_size=16, stride=1)),  # 32, 24, 24
            ("bano2", nn.BatchNorm1d(32)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool1d(kernel_size=2, stride=2)),

            ("conv3", nn.Conv1d(32, 64, kernel_size=5, stride=1)),  # 32, 24, 24
            ("bano3", nn.BatchNorm1d(64)),
            ("relu3", nn.ReLU(inplace=True)),
            ("pool3", nn.MaxPool1d(kernel_size=2, stride=2)),

            ("conv4", nn.Conv1d(64, 64, kernel_size=5, stride=1)),  # 32, 24, 24
            ("bano4", nn.BatchNorm1d(64)),
            ("relu4", nn.ReLU(inplace=True)),
            ("pool4",  nn.AdaptiveMaxPool1d(4)),

        ]))


        self.optimizer = optimizer(list(self.features.parameters()), lr=lr,weight_decay=weight_decay)
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight,0.1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

# Classifier
class task_classifier(nn.Module):
    def __init__(self, hidden_size, optimizer, lr, weight_decay, class_num=7):
        super(task_classifier,self).__init__()
        self.task_classifier = nn.Sequential()
        self.task_classifier.add_module('t1_fc1', nn.Linear(hidden_size, hidden_size))
        self.task_classifier.add_module('t1_fc2', nn.Linear(hidden_size, class_num))
        self.optimizer = optimizer(self.task_classifier.parameters(),
                                   lr=lr, weight_decay=weight_decay)

    def initialize_paras(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight,a=0,mode='fan-out')
            elif isinstance(layer,torch.nn.Linear):
                init.kaiming_normal_(layer.weight)
            elif isinstance(layer,torch.nn.BatchNorm2d) or isinstance(layer,torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = torch.flatten(x, 1)
        y = self.task_classifier(x)
        return y

