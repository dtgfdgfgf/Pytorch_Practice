"""
Define and train a feedforward neural network model 
by downloading the data set from Kaggle, processing the data
"""

"""
import kaggle

Download a specific dataset
The parameter format is 'dataset-owner/dataset-name'
kaggle.api.authenticate()
#kaggle.api.dataset_download_files('kmalit/bank-customer-churn-prediction', unzip=True)
kaggle.api.dataset_download_files('shantanudhakadd/bank-customer-churn-prediction', path='C:/Users/USER/Desktop/pytorch_class/G_D/kaggle_dataset', unzip=True)

"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import scipy.stats as stats

def createNet(nlayer, nunit, BN):
    class Net(nn.Module):       
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()
            self.bnorms = nn.ModuleDict()
            self.nlayers = nlayer
            self.layers['input'] = nn.Linear(2, nunit)
            for i in range(nlayer):
                self.bnorms[f'bnorm{i}'] = nn.BatchNorm1d(nunit)
                self.layers[f'hidden{i}'] = nn.Linear(nunit, nunit)
            self.layers['output'] = nn.Linear(nunit, 1)

        def forward(self, x):
            x = F.relu(self.layers['input'](x))
                
            if BN:
                for i in range(self.nlayers):
                    x = self.bnorms[f'bnorm{i}'](x)
                    x = self.layers[f'hidden{i}'](x)
                    x = F.relu(x)
                x = self.layers['output'](x)   
                
            else:
                for i in range(self.nlayers):
                    #self.bnorms[f'bnorm{i}'](x)
                    x = self.layers[f'hidden{i}'](x)
                    x = F.relu(x)
                x = self.layers['output'](x)   
            return x
        
    
    return Net()
    
def Train(net, optimizerAlgo, LR, L2):
    
    epochs = 300
    lossfun = nn.MSELoss()
    optifun = getattr(torch.optim, optimizerAlgo)
    optimizer = optifun(net.parameters(), lr=LR)

    trainLoss = torch.zeros(epochs)
    testLoss = torch.zeros(epochs)
    for epochi in range(epochs):
        
        net.train()
        
        #batchAcc = []
        batchLoss = []
        
        for X, y in train_loader:
            # forward
            yHat = net(X)
            #print(f'yHat={yHat}, y={y}')
            loss = lossfun(yHat, y)
            #print(f'yHat={yHat}, y={y}, loss={loss}')
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss from this batch
            batchLoss.append(loss.item())
                        
        trainLoss[epochi] = np.mean(batchLoss)
        print (trainLoss[epochi])
        
        net.eval()
        X, y = next(iter(test_loader))
        with torch.no_grad():
            yHat = net(X)
        testLoss[epochi] = lossfun(yHat, y).item()
        
        print(f'epoch{epochi}', '\n')

    return trainLoss, testLoss

path='C:/Users/USER/Desktop/pytorch_class/G_D/kaggle_dataset/Churn_Modelling.csv'
data = pd.read_csv(path)

# 先取出要用的值

#dataT = data[['Age']]
#labelsT = data[['Age']]

#dataT = data.drop(['Age', 'RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender', 'Exited'],axis=1)

# z-score
data = data[data['Balance']>0]
label = data['Age']

droped_key = data.keys().drop(['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender', 'Exited', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'CreditScore', 'Tenure'])
data[droped_key] = data[droped_key].apply(stats.zscore)
#print(data['Balance'])
train_key = droped_key.drop('Age')

# 轉tensor
tensor_train_data = torch.tensor(data[train_key].values).float()
label = torch.tensor(data['Age'].values).float()
#label = torch.tensor(label.values).float()
label = label[:,None]

# split the data
train_data, test_data, train_labels, test_labels = train_test_split(tensor_train_data, label, test_size=0.1)

# convert into PyTorch Datasets
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

# translate into dataloader objects
batchsize = 64
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0], shuffle=False, drop_last=False)

# print(train_data)

# define metaparameters
numlayer = 4
numunits = 64
bn = True
lr = 0.005
L2 = 0

net = createNet(numlayer, numunits, bn)
train_loss, test_loss = Train(net, 'SGD', lr, L2)

print(train_loss, test_loss)

plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()












