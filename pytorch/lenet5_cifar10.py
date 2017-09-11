import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

DATA_PATH = '/data/vision/fisher/data1/vsmolyakov/pytorch/'

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,(5,5))
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,(5,5))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
       
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

use_gpu = torch.cuda.is_available()

#load data
print "loading data..."
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#instantiate CNN
net = Net()
if use_gpu:
    print "found CUDA GPU..."
    net = net.cuda()

print net

#define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print "training..."
for epoch in range(16):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)


        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 500 == 0:
            print "epoch: %4d, mini-batch: %4d, loss: %.3f" %(epoch+1, i, running_loss / 500.0)
            running_loss = 0.0
       
print "finished training..."

correct = 0
total = 0
for data in testloader:
    images, labels = data
    if use_gpu:
        outputs = net(Variable(images.cuda()))
    else:
        outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    if use_gpu:
        correct += (predicted.cpu() == labels).sum()
    else:
        correct += (predicted == labels).sum()


print "Test accuracy: %2.2f" %(100.0 * correct / total)


