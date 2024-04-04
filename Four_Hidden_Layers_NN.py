import torch
import torch.nn as nn


class My_NN_4(nn.Module):
  def __init__(self, li, lh1, lh2, lh3, lh4, lo):
    super().__init__()
    self.li = li
    self.linear1 = nn.Linear(li, lh1)
    self.linear2 = nn.Linear(lh1, lh2)
    self.linear3 = nn.Linear(lh2, lh3)
    self.linear4 = nn.Linear(lh3, lh4)
    self.linear5 = nn.Linear(lh4, lo)


  def forward(self,x):
    x = x.view(-1, self.li)
    x = self.linear1(x)
    x = torch.sigmoid(x)

    x = self.linear2(x)
    x = torch.sigmoid(x)

    x = self.linear3(x)
    x = torch.sigmoid(x)

    x = self.linear4(x)
    x = torch.sigmoid(x)

    x = self.linear5(x)
    return x
