"""
The main purpose of this program is to compare model accuracy and loss changes under different settings 
by adjusting the learning rate and using/not using batch regularization. T
his helps understand the impact of learning rate and batch regularization on model training
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class ANNmodel(nn.Module):
    def __init__(self, nlayer, nunit):
        super().__init__()
        
        self.layers = nn.ModuleDict()
        self.bnorms = nn.ModuleDict()
        self.nlayers = nlayer
        self.layers['input'] = nn.Linear(2, nunit)
        for i in range(nlayer):
            self.layers[f'hidden{i}'] = nn.Linear(nunit,nunit)
            self.bnorms[f'bnorm{i}'] = nn.BatchNorm1d(nunit)
        self.layers['output'] = nn.Linear(nunit, 1)
        
    def forward(self, x, BN):
        x = F.relu(self.layers['input'](x))
        
        if BN:
            for i in range(self.nlayers):
                x = self.bnorms[f'bnorm{i}'](x)
                x = self.layers[f'hidden{i}'](x)
                x = F.relu(x)
            x = self.layers['output'](x)
            
        else:
            for i in range(self.nlayers):
                #x = self.bnorms[f'bnorm{i}'](x)
                x = self.layers[f'hidden{i}'](x)
                x = F.relu(x)
            x = self.layers['output'](x)
        return x
        

def trainTheModel(theModel, LR, BN):

  # define the loss function and optimizer
    numepochs = 1000
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(theModel.parameters(),lr=LR)
    losses = torch.zeros(numepochs)
    
    # loop over epochs
    for epochi in range(numepochs):
      
        # forward pass
        yHat = theModel(data, BN)
        
        # compute loss
        loss = lossfun(yHat,labels)
        losses[epochi] = loss
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
           
    # final forward pass to get accuracy
    predictions = theModel(data, BN)
    #predlabels = torch.argmax(predictions,axis=1)
    acc = 100*torch.mean(((predictions>0.5) == labels).float())
      
    # total number of trainable parameters in the model
    nParams = sum(p.numel() for p in theModel.parameters() if p.requires_grad)
      
    # function outputs
    return acc, nParams, losses     
        
nPerClust = 100
blur = 1

A = [  1,  3 ]
B = [  1, -2 ]

# generate data
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]

# true labels
labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1))))

# concatanate into a matrix
data_np = np.hstack((a,b)).T

#z-score
data_np_zscore = (data_np - data_np.mean(axis=0)) / data_np.std(axis=0, ddof=1)

# convert to a pytorch tensor
data = torch.tensor(data_np_zscore).float()
labels = torch.tensor(labels_np).float()

# learning rates
learningrates = np.linspace(.005,0.2,50)

# initialize
numepochs = 1000

accByLR = []
allLosses = np.zeros((len(learningrates),numepochs))

BN_accByLR = []
BN_allLosses = np.zeros((len(learningrates),numepochs))

# the loop
for i,lr in enumerate(learningrates):

    # create and run the model
    #ANNclassify,lossfun,optimizer = createANNmodel(lr)
    ann = ANNmodel(3, 16)
    acc, nParams, losses = trainTheModel(ann, lr, False)
    
    # store the results
    accByLR.append(acc)
    allLosses[i,:] = losses.detach()
    
    BN_ann = ANNmodel(3, 16)
    BN_acc, BN_nParams, BN_losses = trainTheModel(ann, lr, True)
    
    # store the results
    BN_accByLR.append(acc)
    BN_allLosses[i,:] = BN_losses.detach()

# plot the results
fig,ax = plt.subplots(2,2,figsize=(16,16))

ax[0][0].plot(learningrates,accByLR,'s-')
ax[0][0].set_xlabel('Learning rate')
ax[0][0].set_ylabel('Accuracy')
ax[0][0].set_title('Accuracy by learning rate')

ax[0][1].plot(allLosses.T)
ax[0][1].set_title('Losses by learning rate')
ax[0][1].set_xlabel('Epoch number')
ax[0][1].set_ylabel('Loss')

ax[1][0].plot(learningrates,BN_accByLR,'s-')
ax[1][0].set_xlabel('Learning rate')
ax[1][0].set_ylabel('BN_Accuracy')
ax[1][0].set_title('BN_Accuracy by learning rate')

ax[1][1].plot(BN_allLosses.T)
ax[1][1].set_title('BN_Losses by learning rate')
ax[1][1].set_xlabel('Epoch number')
ax[1][1].set_ylabel('BN_Loss')
plt.show()



