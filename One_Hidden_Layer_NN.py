import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models,transforms
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary


class My_NN(nn.Module):  # 1 hidden layer Neural Network
    def __init__(self, li, lh, lo):
        super().__init__()

        self.layer1 = nn.Linear(li, lh)
        self.layer2 = nn.Linear(lh, lo)
        self.li = li

    def forward(self, x):
        x = x.reshape(x.shape[0], self.li)

        x = self.layer1(x)
        x = torch.sigmoid(x)

        x = self.layer2(x)

        return x
