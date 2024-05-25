"""
The main purpose of this code is to explore the impact of different batch sizes 
on the training and testing performance of neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

# create ANN model

def createANewModel():
    
    # model architecture
    ANNiris = nn.Sequential(
        nn.Linear(11,64),   # input layer
        nn.ReLU(),         # activation unit
        nn.Linear(64,64),  # hidden layer
        nn.ReLU(),         # activation unit
        nn.Linear(64,1),   # output units
          )
    
    # loss function
    lossfun = nn.BCEWithLogitsLoss()
    
    # optimizer
    optimizer = torch.optim.SGD(ANNiris.parameters(),lr=0.1)
    
    return ANNiris,lossfun,optimizer

# global parameter
numepochs = 1000

# train the model
def trainTheModel():
    
    # initialize accuracies as empties
    trainAcc = []
    testAcc  = []
    losses   = []
    
    # loop over epochs
    for epochi in range(numepochs):
        
        # loop over batches
        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:
            
            ANNiris.train()  
            
            # forward
            yHat = ANNiris(X)
            loss = lossfun(yHat,y)
        
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # compute training accuracy for this batch
            train_probs = torch.sigmoid(yHat)
            train_predicted_labels = (train_probs > 0.5).float()
            batchAcc.append(100 * torch.mean((train_predicted_labels == y).float()).item())
            batchLoss.append(loss.item())
        # end of batch loop
      
        # get average training accuracy
        trainAcc.append( np.mean(batchAcc) )
        losses.append( np.mean(batchLoss) )
        
        ANNiris.eval()
      
        # test accuracy
        X,y = next(iter(test_loader))
        with torch.no_grad():
            yHat = ANNiris(X)
            test_probs = torch.sigmoid(yHat)
        test_predicted_labels = (test_probs > 0.5).float()
        testAcc.append(100 * torch.mean((test_predicted_labels == y).float()).item())
    
    # function output
    return trainAcc,testAcc,losses

# import dataset
import pandas as pd
import seaborn as sns
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

data = pd.read_csv(url,sep=';')
cols2zscore = data.keys()
cols2zscore = cols2zscore.drop('quality')

# z-score
for col in cols2zscore:
    meanval   = np.mean(data[col])
    stdev     = np.std(data[col],ddof=1)
    data[col] = (data[col]-meanval) / stdev
    
# create a new column for binarized quality
data['boolQuality'] = 0
# data['boolQuality'][data['quality']<6] = 0
data['boolQuality'][data['quality']>5] = 1

dataT  = torch.tensor( data[cols2zscore].values ).float()
labels = torch.tensor( data['boolQuality'].values ).float()
labels = labels[:,None]
#print( dataT.shape )
#print( labels.shape )

# split the data
train_data,test_data, train_labels,test_labels = train_test_split(dataT, labels, test_size=0.2)

# convert into PyTorch Datasets
train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

# translate into dataloader objects
iterations = 5
trainAccList = []
testAccList = []
for n in range(iterations):
    batchsize = 2 ** (2*n+1)
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0]) # how big should these batches be??
    
    # create a model
    ANNiris,lossfun,optimizer = createANewModel()
    
    # train the model
    trainAcc,testAcc,losses = trainTheModel()
    trainAccList.append(trainAcc)
    testAccList.append(testAcc)

# plot the results

trainArr = np.array(trainAccList)
testArr = np.array(testAccList)
fig,ax = plt.subplots(1,2,figsize=(17,8))

batch_sizes = [2 ** (2*n+1) for n in range(iterations)]
for idx, trainAcc in enumerate(trainArr):
    ax[0].plot(trainAcc, label=f'Batch size {batch_sizes[idx]}')
ax[0].set_ylabel('Acc')
ax[0].set_xlabel('Epochs')
ax[0].set_title('Train Acc')
ax[0].legend()

for idx, testAcc in enumerate(testArr):
    ax[1].plot(testAcc, label=f'Batch size {batch_sizes[idx]}')
ax[1].set_ylabel('Acc')
ax[1].set_xlabel('Epochs')
ax[1].set_title('Test Acc')
ax[1].legend()

plt.tight_layout()
plt.show()

