#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implement deep q-network based agent to solve shepherding task

The neural network is implemented using the pytorch library
"""

# core modules
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# define Q-network that outputs q-value for each action
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        # 3 fully connected layers
        self.linear1 = nn.Linear(10,256)
        self.linear2 = nn.Linear(256,64)
        self.linear3 = nn.Linear(64,8)

    def forward(self, x):
        # perform forward computation
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x

