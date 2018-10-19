import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
torch.manual_seed(0)

def p_k(x):
    #target density: p(x) = 1/Z exp(-E(x))
    #TODO: higher dimensional distribution
    #TODO: let p_k(x) be graphical model posterior
    x1, x2 = torch.chunk(x, chunks=2, dim=1)
    norm = torch.sqrt(x1**2 + x2**2)

    exp1 = torch.exp(-0.5 * ((x1 - 2) / 0.6)**2)
    exp2 = torch.exp(-0.5 * ((x1 + 2) / 0.6)**2)
    u = 0.5 * ((norm - 2)/0.4)**2 - torch.log(exp1 + exp2 + 1e-8)

    return torch.exp(-u)

class FreeEnergyLoss(nn.Module):

    def __init__(self, density):
        super(FreeEnergyLoss, self).__init__()
        self.density = density

    def forward(self, xk, logdet_jacobians):
        logdet_jacobians_sum = sum(logdet_jacobians)
        return (-logdet_jacobians_sum - torch.log(self.density(xk) + 1e-8)).mean()

class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        
        activation = F.linear(x, self.weight, self.bias)
        return x + self.scale * self.tanh(activation)

class PlanarFlow_LogDetJacobian(nn.Module):

    def __init__(self, planar):
        super(PlanarFlow_LogDetJacobian, self).__init__()
        
        self.weight = planar.weight
        self.bias = planar.bias
        self.scale = planar.scale
        self.tanh = planar.tanh

    def forward(self, x):
        activation = F.linear(x, self.weight, self.bias)
        psi = (1 - self.tanh(activation)**2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-8) 


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super(NormalizingFlow, self).__init__()

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))

        self.logdet_jacobians = nn.Sequential(*(
            PlanarFlow_LogDetJacobian(t) for t in self.transforms
        ))
       
    def forward(self, x):
      
        logdet_jacobians_output = []
        for transform, logdet_jacobian in zip(self.transforms, self.logdet_jacobians):
            logdet_jacobians_output.append(logdet_jacobian(x))  #forward call on prev sample
            x = transform(x)
        
        xk = x
        
        return xk, logdet_jacobians_output

use_gpu = torch.cuda.is_available()

#instantiate NF 
net = NormalizingFlow(dim=2, flow_length=8)
if use_gpu:
    print "found CUDA GPU..."
    net = net.cuda()
print net

#define loss and optimizer
criterion = FreeEnergyLoss(density=p_k) 
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.RMSprop(net.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=4, gamma=0.5) #half learning rate every 4 epochs

#training parameters
xdim = 2
num_iter = 16 
batch_size = 512 
generate_plots = True

print "training..."
training_loss_tot = []
learning_rate_schedule = []
for epoch in tqdm(range(16)):

    scheduler.step()
    running_loss = 0.0
    for iteration in range(num_iter):

        data = torch.zeros(batch_size, xdim).normal_(mean=0, std=1)
    
        if use_gpu:
            data = Variable(data.cuda())
        else:
            data = Variable(data)

        optimizer.zero_grad()
        xk, logdet_jacobians = net(data)
        loss = criterion(xk, logdet_jacobians)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0].cpu()
    #end for
    training_loss_tot.append(running_loss / float(num_iter))
    learning_rate_schedule.append(scheduler.get_lr())
    print "epoch: %4d, loss: %.3f" %(epoch+1, running_loss / float(num_iter))

    if (generate_plots):
        samples = torch.zeros(1000, xdim).normal_(mean=0, std=1)

        if use_gpu:
            samples = Variable(samples.cuda())
        else:
            samples = Variable(samples)

        xk, logdetj = net(samples)

        plt.figure()
        plt.scatter(xk.data.cpu().numpy()[:,0], xk.data.cpu().numpy()[:,1])
        plt.title('epoch: ' + str(epoch))
        plt.savefig('./figures/nf_epoch_' + str(epoch) + '.png')
#end for
       
print "finished training..."

#generate plots

#plot original density
x1 = np.linspace(-5, 5, 300)
x2 = np.linspace(-5, 5, 300)
x1, x2 = np.meshgrid(x1, x2)
shape = x1.shape
x1 = x1.ravel()
x2 = x2.ravel()

xt = np.c_[x1, x2]
xt = torch.FloatTensor(xt)
xt = Variable(xt)
gt_density = p_k(xt).data.cpu().numpy().reshape(shape)

plt.figure()
plt.imshow(gt_density, cmap='summer')
plt.savefig('./figures/nf_ground_truth.png')

#plot training loss
plt.figure()
plt.plot(training_loss_tot, label='RMSProp')
plt.title("Normalizing Flow Training Loss")
plt.xlabel("Epoch"); plt.ylabel("Free Energy Loss")
plt.legend(); plt.grid(True);
plt.savefig('./figures/nf_training_loss.png')

#plot learning rate schedule
plt.figure()
plt.plot(learning_rate_schedule, label='learning rate')
plt.title("NF learning rate schedule")
plt.xlabel("Epoch"); plt.ylabel("Learning Rate")
plt.legend(); plt.grid(True)
plt.savefig('./figures/nf_lr_schedule.png')


