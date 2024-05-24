"""
Observe the impact of different learning rates on model accuracy
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
        self.nlayers = nlayer
        self.layers['input'] = nn.Linear(2, nunit)
        for i in range(nlayer):
            self.layers[f'hidden{i}'] = nn.Linear(nunit,nunit)
        self.layers['output'] = nn.Linear(nunit, 1)
        
    def forward(self, x):
        x = F.relu(self.layers['input'](x))
        for i in range(self.nlayers):
            x = F.relu(self.layers[f'hidden{i}'](x))
        x = self.layers['output'](x)
        return x
        

def trainTheModel(theModel, LR):

  # define the loss function and optimizer
    numepochs = 1000
    lossfun = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(theModel.parameters(),lr=LR)
    losses = torch.zeros(numepochs)
    
    # loop over epochs
    for epochi in range(numepochs):
      
        # forward pass
        yHat = theModel(data)
        
        # compute loss
        loss = lossfun(yHat,labels)
        losses[epochi] = loss
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
      
      
    # final forward pass to get accuracy
    predictions = theModel(data)
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

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# learning rates
learningrates = np.linspace(.05,.2,50)

# initialize
numepochs = 1000
accByLR = []
allLosses = np.zeros((len(learningrates),numepochs))


# the loop
for i,lr in enumerate(learningrates):

    # create and run the model
    #ANNclassify,lossfun,optimizer = createANNmodel(lr)
    ann = ANNmodel(3, 5)
    acc, nParams, losses = trainTheModel(ann, lr)
    
    # store the results
    accByLR.append(acc)
    allLosses[i,:] = losses.detach()

# plot the results
fig,ax = plt.subplots(1,2,figsize=(16,4))

ax[0].plot(learningrates,accByLR,'s-')
ax[0].set_xlabel('Learning rate')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Accuracy by learning rate')

ax[1].plot(allLosses.T)
ax[1].set_title('Losses by learning rate')
ax[1].set_xlabel('Epoch number')
ax[1].set_ylabel('Loss')
plt.show()



