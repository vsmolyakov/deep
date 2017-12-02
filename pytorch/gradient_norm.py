import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable

from tqdm import tqdm
from sklearn.datasets import load_boston

np.random.seed(0)
#torch.manual_seed(0)

#load data
boston = load_boston()
X_data = boston.data
y_data = boston.target

train_data = []
for idx in range(X_data.shape[0]):
    sample = {}
    X_sample = X_data[idx,:]
    y_sample = np.array([y_data[idx]])
    sample['X'] = torch.FloatTensor(X_sample)
    sample['y'] = torch.FloatTensor(y_sample)
    train_data.append(sample)
#end for

#simple example
x = Variable(torch.randn(1,1), requires_grad = True)
y = 3 * x
z = y**2
z.backward()

#model parameters
dnn_input_dim = X_data.shape[1] 
dnn_output_dim = 1 
weight_decay = 1e-3 
learning_rate = 1e-3 

#training parameters
num_epochs = 128 
batch_size = 128 

#MLP
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim  
 
        self.fc1 = nn.Linear(self.input_dim, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, self.output_dim)
        #self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = self.softmax(x)
        return x 

model = DNN(dnn_input_dim, dnn_output_dim)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()

print model 

#define loss and optimizer
criterion = nn.MSELoss(size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)

training_loss = []

grad_norm_df = pd.DataFrame()
weight_norm_df = pd.DataFrame()

print "training..."
for epoch in range(num_epochs):
    
    running_train_loss = 0.0
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4, 
        drop_last = True)
        
    model.train()
        
    for bidx, batch in enumerate(tqdm(train_data_loader)):
     
        x_data = Variable(batch['X'])
        y_true = Variable(batch['y'])

        if use_gpu:
            x_data, y_true = x_data.cuda(), y_true.cuda()
        
        optimizer.zero_grad()

        y_pred = model(x_data)

        loss = criterion(y_pred, y_true) 
        loss.backward()
        optimizer.step()
       
        #gradient and weight norms 
        df_idx = bidx + epoch * batch_size
        grad_norm_df.loc[df_idx, 'df_idx'] = df_idx
        weight_norm_df.loc[df_idx, 'df_idx'] = df_idx

        grad_norm_df.loc[df_idx,'w1_grad_l2'] = torch.norm(model.fc1.weight.grad, 2).cpu().data[0] 
        weight_norm_df.loc[df_idx,'w1_data_l2'] = torch.norm(model.fc1.weight, 2).cpu().data[0]

        grad_norm_df.loc[df_idx,'w2_grad_l2'] = torch.norm(model.fc2.weight.grad, 2).cpu().data[0] 
        weight_norm_df.loc[df_idx,'w2_data_l2'] = torch.norm(model.fc2.weight, 2).cpu().data[0] 

        grad_norm_df.loc[df_idx,'w3_grad_l2'] = torch.norm(model.fc3.weight.grad, 2).cpu().data[0] 
        weight_norm_df.loc[df_idx,'w3_data_l2'] = torch.norm(model.fc3.weight, 2).cpu().data[0] 

        running_train_loss += loss.cpu().data[0]        
    #end for

    training_loss.append(running_train_loss)
    print "epoch: %4d, training loss: %.4f" %(epoch+1, running_train_loss)
#end for

#generate plots
plt.figure()
plt.plot(training_loss, c='b', label='Adam')
plt.title('MLP training loss')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig('../figures/mlp_training_loss.png')

plt.figure()
plt.plot(grad_norm_df['w1_grad_l2'], c='r', alpha=0.8, lw=2.0, label='w1 grad l2')
plt.plot(grad_norm_df['w2_grad_l2'], c='b', alpha=0.8, label='w2 grad l2')
plt.plot(grad_norm_df['w3_grad_l2'], c='k', alpha=0.8, label='w3 grad l2')
plt.title('MLP gradient norm')
plt.xlabel('num batches')
plt.ylabel('l2 norm')
plt.legend()
plt.savefig('../figures/mlp_gradient_norm.png')

plt.figure()
plt.plot(weight_norm_df['w1_data_l2'], c='r', alpha=0.8, label='w1 data l2')
plt.plot(weight_norm_df['w2_data_l2'], c='b', alpha=0.8, label='w2 data l2')
plt.plot(weight_norm_df['w3_data_l2'], c='k', alpha=0.8, label='w3 data l2')
plt.title('MLP weight norm')
plt.xlabel('num batches')
plt.ylabel('l2 norm')
plt.legend()
plt.savefig('../figures/mlp_weight_norm.png')


