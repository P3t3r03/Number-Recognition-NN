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
import wandb

from Initialise_dataset import mnist
from One_Hidden_Layer_NN import My_NN
from Four_Hidden_Layers_NN import My_NN_4
from CNN import CNN

sweep_config = {
      "method": "random",
      "name": "sweep",
      "architecture": "NN",
      "dataset": "MNIST",
      "metric": {"goal":"maximize", "name": "val_acc"},
      "parameters": {
        "batch_size": {"values": [16,32,64]},
        "epochs": {"values":[5,10,15]},
        "lr": {"max": 0.1, "min": 0.0001},
      },
}


sweep_id = wandb.sweep(sweep=sweep_config, project="sweep_test_CNN")



# train_loader, valid_loader, test_loader=mnist(batch_sz)  # Load all the data into train, validation, test sets


# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="number-recognition",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.05,
#     "architecture": "NN",
#     "dataset": "MNIST",
#     "epochs": 10,
#     }
# )
# Test to see if we're loading the data correctly and visualizing the image
# batch = next(iter(train_loader))
# plt.imshow(batch[0][52,0,:,:], cmap = "gray")
# plt.title(batch[1][52])
# plt.show()
'''

device = torch.device("cpu")  # Selecting Device you can use cuda:0 if you have a gpu
li = 28*28  # size of the image we're loading
lh = 100  # Can change this around size of hidden layer
lo = 10  # 10 output classes when classifying from 0-9

net = My_NN(li, lh, lo).to(device)  # Initialises the Neural Network and passes it to the cpu/gpu

# Hyperparameters for learning
num_epochs = 10
learning_rate = 0.05

opt = optim.SGD(net.parameters(), lr = learning_rate)  # Initialises Optimiser that uses
# Stochastic Gradient Descent to optimise the net parameters as training happens
ls = []  # list of losses

# Training Loop
for i in range(num_epochs):
  total_loss = 0  # Loss per Epoch
  # Batch Loop
  ys = []
  yhats = []
  for batch in train_loader:
    X, y = batch[0].to(device), batch[1].to(device) # Sends Batch to Device to operate on
    yhat = net.forward(X)  # forward pass called in Neural Network
    loss = F.cross_entropy(yhat, y)  # Loss calculated for this batch
    total_loss += loss
    opt.zero_grad()  # Zeroes out Network parameter gradients so that we have a fresh start
    # Backpropagation loss
    loss.backward()  # Calculates new gradients for the parameters
    opt.step()  # Updates the parameters based on the gradients and learning rate

   # Appends the loss for the Epoch.
ls_cpu = [loss.cpu().item() for loss in ls]  # Transfers the data back to the cpu if using gpu earlier
for batch in valid_loader:
  X,y = batch[0].to(device), batch[1].to(device)
  yhat = net(X)
  labels = torch.argmax(yhat, axis=1)
  ys.extend(y.cpu().numpy())
  yhats.extend(labels.cpu().numpy())

ys = np.array(ys)
yhats = np.array(yhats)
num_all_test_samples = len(ys)
num_correct = np.sum(ys == yhats)
val_acc = num_correct / num_all_test_samples
ls.append(total_loss/len(train_loader))
wandb.log({"loss": loss, "val_acc": val_acc})
'''
def train_model(train_loader, opt, device, net):
  train_loss = 0
  ys = []
  yhats = []
  for batch in train_loader:
    X, y = batch[0].to(device), batch[1].to(device) # Sends Batch to Device to operate on
    yhat = net.forward(X)  # forward pass called in Neural Network
    loss = F.cross_entropy(yhat, y)  # Loss calculated for this batch
    train_loss += F.cross_entropy(yhat, y)
    opt.zero_grad()  # Zeroes out Network parameter gradients so that we have a fresh start
    # Backpropagation loss
    loss.backward()  # Calculates new gradients for the parameters
    opt.step()  # Updates the parameters based on the gradients and learning rate
    labels = torch.argmax(yhat, axis=1)
    ys.extend(y.cpu().numpy())
    yhats.extend(labels.cpu().numpy())


  ys = np.array(ys)
  yhats = np.array(yhats)
  num_all_test_samples = len(ys)
  num_correct = np.sum(ys == yhats)
  train_acc = num_correct / num_all_test_samples
 #  train_acc = train_acc / len(train_loader)
  return train_acc, train_loss

def evaluate_model(valid_loader, device, net):
  ys = []
  yhats = []
  for batch in valid_loader:
    X,y = batch[0].to(device), batch[1].to(device)
    yhat = net.forward(X)
    val_loss = F.cross_entropy(yhat, y)
    labels = torch.argmax(yhat, axis=1)
    ys.extend(y.cpu().numpy())
    yhats.extend(labels.cpu().numpy())
  ys = np.array(ys)
  yhats = np.array(yhats)
  num_all_test_samples = len(ys)
  num_correct = np.sum(ys == yhats)
  val_acc = num_correct / num_all_test_samples
  return val_acc, val_loss
def main():
  run = wandb.init()

  lr = wandb.config.lr
  batch_sz = wandb.config.batch_size
  epochs = wandb.config.epochs
  train_loader, valid_loader, test_loader=mnist(batch_sz)

  device = torch.device("cpu")  # Selecting Device you can use cuda:0 if you have a gpu
  li = 28*28  # size of the image we're loading
  lh = 100  # Can change this around size of hidden layer
  lo = 10  # 10 output classes when classifying from 0-9

  # net = My_NN(li, lh, lo).to(device)  # Initialises the Neural Network and passes it to the cpu/gpu
  net = CNN()
  opt = optim.SGD(net.parameters(), lr = lr)

  for i in range(epochs):
    train_acc, train_loss = train_model(train_loader, opt, device, net)

    val_acc, val_loss = evaluate_model(valid_loader, device, net)

    wandb.log({
      "epoch": i,
      "train_acc": train_acc,
      "train_loss": train_loss,
      "val_acc": val_acc,
      "val_loss": val_loss
    })

wandb.agent(sweep_id, function=main, count = 100)
# Plots Loss
# plt.plot(ls_cpu)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
#plt.show()

# # Initilizing true and predicted classes
# ys = []
# yhats = []

# # Flag so that the computation is more efficient as we don't have to calculate the gradients as we're testing.
# with torch.no_grad():
#   # Retriving and calculating true and predicted classes
#   for batch in test_loader:
#     X, y = batch[0].to(device), batch[1].to(device)  # Sends the true and predicted classes to the device chosen(cpu/gpu)
#     yhat = net(X)  # No need to call forward as it has been overridden in class definition when using PyTorch
#     labels = torch.argmax(yhat, axis = 1)  # Labels the sample with the class it has the highest probability of being
#     # Extends the True and predicted class list
#     ys.extend(y.cpu().numpy())
#     yhats.extend(labels.cpu().numpy())

# #Convert to np array
# ys = np.array(ys)
# yhats = np.array(yhats)

# # print(ys)
# # print(yhats)
# # print(ys.shape)

# # compute the test accuracy
# num_all_test_samples = len(ys)
# num_correct = np.sum(ys == yhats)
# print(num_correct/ num_all_test_samples)

# net = My_NN(784, 100, 10)
# for params in net.parameters():
#   print(params.shape)

# for batch in test_loader:
#   print(batch)
#   break


# Visualize Batch
# plt.figure(figsize=(16,16))
# for i in range(64):
#   plt.subplot(8,8, i+1)
#   plt.imshow(batch[0][i, 0,:,:])
#   plt.title(batch[1][i].item())
#   plt.axis('off')
# plt.show()


# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 10,
#     }
# )

# device = torch.device("cpu")
# li = 28*28
# lh1 = 128
# lh2 = 128
# lh3 = 64
# lh4 = 64
# lo = 10

# net_4 = My_NN_4(li, lh1, lh2, lh3, lh4, lo).to(device)
# num_epochs = 200 # Takes much longer to run but seems to be plateau until 60~70ish epochs then decreasing loss.
# lr = 0.1

# opt = optim.SGD(net_4.parameters(), lr=lr)
# ls = []
# vls = []

# # Same as 1 hidden layer
# for i in range(num_epochs):
#   total_ls = 0
#   for batch in train_loader:
#     X, y = batch[0].to(device), batch[1].to(device)
#     #doing forward pass through the network
#     yhat = net_4.forward(X)
#     #calculating loss
#     loss = F.cross_entropy(yhat, y)
#     #emptyting hte optimizer buffes to store gradients
#     opt.zero_grad()
#     #calculating gradients
#     loss.backward()
#     #taking a step in the negative gradient direction
#     opt.step()
#     total_ls += loss

#   with torch.no_grad():
#     val_ls = 0
#     for batch in valid_loader:
#       X, y = batch[0].to(device), batch[1].to(device)
#       yhat = net_4.forward(X)
#       # calculating loss
#       loss = F.cross_entropy(yhat, y)
#       val_ls += loss

#   ls.append(total_ls/len(train_loader))
#   vls.append(val_ls/len(valid_loader))

# ls_cpu = [loss.cpu().item() for loss in ls]
# vls_cpu = [loss.cpu().item() for loss in vls]
# plt.plot(ls_cpu, 'b', label='Training Loss')
# plt.plot(vls_cpu, 'g', label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# ys = []
# yhats = []

# # Flag so that the computation is more efficient as we don't have to calculate the gradients as we're testing.
# with torch.no_grad():
#   # Retriving and calculating true and predicted classes
#   for batch in test_loader:
#     X, y = batch[0].to(device), batch[1].to(device)  # Sends the true and predicted classes to the device chosen(cpu/gpu)
#     yhat = net_4(X)  # No need to call forward as it has been overridden in class definition when using PyTorch
#     labels = torch.argmax(yhat, axis = 1)  # Labels the sample with the class it has the highest probability of being
#     # Extends the True and predicted class list
#     ys.extend(y.cpu().numpy())
#     yhats.extend(labels.cpu().numpy())

# #Convert to np array
# ys = np.array(ys)
# yhats = np.array(yhats)

# print(ys)
# print(yhats)
# print(ys.shape)

# # compute the test accuracy
# num_all_test_samples = len(ys)
# num_correct = np.sum(ys == yhats)
# print(num_correct/ num_all_test_samples)

