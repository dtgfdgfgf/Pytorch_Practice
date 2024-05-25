"""
Test optimizers and L2
"""

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# create data

nPerClust = 300
blur = 1

A = [ 1, 1 ]
B = [ 5, 1 ]
C = [ 4, 3 ]

# generate data
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]
c = [ C[0]+np.random.randn(nPerClust)*blur , C[1]+np.random.randn(nPerClust)*blur ]

# true labels
labels_np = np.hstack((  np.zeros((nPerClust)),
                         np.ones( (nPerClust)),
                       1+np.ones( (nPerClust))  ))

# concatanate into a matrix
data_np = np.hstack((a,b,c)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).long() # "long" for CrossEntropyLoss

"""
# show the data
fig = plt.figure(figsize=(5,5))
plt.plot(data[np.where(labels==0)[0],0],data[np.where(labels==0)[0],1],'bs',alpha=.5)
plt.plot(data[np.where(labels==1)[0],0],data[np.where(labels==1)[0],1],'ko',alpha=.5)
plt.plot(data[np.where(labels==2)[0],0],data[np.where(labels==2)[0],1],'r^',alpha=.5)
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.ylabel('qwerty dimension 2')
plt.show()
"""

# use scikitlearn to split the data
train_data,test_data, train_labels,test_labels = train_test_split(data, labels, test_size=.1)

# then convert them into PyTorch Datasets (note: already converted to tensors)
train_data = TensorDataset(train_data,train_labels)
test_data  = TensorDataset(test_data,test_labels)

# finally, translate into dataloader objects
batchsize    = 32
train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

# how many batches are there?
print(f'There are {len(train_loader)} batches, each with {batchsize} samples.')

# create a class for the model
def createTheQwertyNet(optimizerAlgo, L2):

    class qwertyNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.input = nn.Linear(2,8)

            self.fc1 = nn.Linear(8,8)

            self.output = nn.Linear(8,3)
      
        # forward pass
        def forward(self,x):
            x = F.relu( self.input(x) )
            x = F.relu( self.fc1(x) )
            return self.output(x)
        
    # create the model instance
    net = qwertyNet()
    
    # loss function
    lossfun = nn.CrossEntropyLoss()
    
    # optimizer
    optifun = getattr( torch.optim,optimizerAlgo )
    optimizer = optifun(net.parameters(),lr=0.001, weight_decay=L2)
    
    return net,lossfun,optimizer

# trains the model

def function2trainTheModel(optimizerType, L2):

    # number of epochs
    numepochs = 50
    
    # create a new model
    net,lossfun,optimizer = createTheQwertyNet(optimizerType, L2)
    
    # initialize losses
    losses   = torch.zeros(numepochs)
    trainAcc = []
    testAcc  = []
    
    # loop over epochs
    for epochi in range(numepochs):
    
        # switch on training mode
        net.train()
      
        # loop over training data batches
        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:
      
            # forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat,y)
        
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # loss from this batch
            batchLoss.append(loss.item())
        
            # compute accuracy
            matches = torch.argmax(yHat,axis=1) == y     # booleans (false/true)
            matchesNumeric = matches.float()             # convert to numbers (0/1)
            accuracyPct = 100*torch.mean(matchesNumeric) # average and x100 
            batchAcc.append(accuracyPct)               # add to list
      
        # get average training accuracy
        trainAcc.append( np.mean(batchAcc) )
      
        # get average losses
        losses[epochi] = np.mean(batchLoss)
      
        # test accuracy
        net.eval()
        X,y = next(iter(test_loader)) 
        with torch.no_grad():
            yHat = net(X)

        testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) ) 
    # end epochs
    
    # function output
    return trainAcc, testAcc, losses,net

# a function that plots the results
def plotTheResults(optimizerType):

    # compute accuracy over entire dataset (train+test)
    yHat = net(data)
    predictions = torch.argmax(yHat,axis=1)
    accuracy = (predictions == labels).float()
    totalAcc = torch.mean(100*accuracy).item()
    
    # and accuracy by group
    accuracyByGroup = np.zeros(3)
    for i in range(3):
        accuracyByGroup[i] = 100*torch.mean(accuracy[labels==i])
    
    
    # create the figure
    fig,ax = plt.subplots(2,2,figsize=(10,6))
    
    # plot the loss function
    ax[0,0].plot(losses.detach())
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_xlabel('epoch')
    ax[0,0].set_title(f'{optimizerType}: Losses')
    
    # plot the accuracy functions
    ax[0,1].plot(learningrates, trainAccList, label='Train')
    ax[0,1].plot(learningrates, testAccList, label='Test')
    ax[0,1].set_xscale('log')
    ax[0,1].set_ylabel('Accuracy (%)')
    ax[0,1].set_xlabel('learningrates')
    ax[0,1].set_title(f'{optimizerType}: Accuracy')
    ax[0,1].legend()
    
    # plot overall accuracy by group
    ax[1,0].bar(range(3),accuracyByGroup)
    #ax[1,0].set_ylim([np.min(accuracyByGroup)-5,np.max(accuracyByGroup)+5])
    ax[1,0].set_xticks([0,1,2])
    ax[1,0].set_xlabel('Group')
    ax[1,0].set_ylabel('Accuracy (%)')
    ax[1,0].set_title(f'{optimizerType}: Accuracy by group')
    
    # scatterplot of correct and incorrect labeled data
    colorShapes = [ 'bs','ko','g^' ] # data markers
    for i in range(3):
        # plot all data points
        ax[1,1].plot(data[labels==i,0],data[labels==i,1],colorShapes[i],
                     alpha=.3,label=f'Group {i}')
        
        # cross-out the incorrect ones
        idxErr = (accuracy==0) & (labels==i)
        ax[1,1].plot(data[idxErr,0],data[idxErr,1],'rx')
    
    ax[1,1].set_title(f'{optimizerType}: Total accuracy: {totalAcc:.2f}%')
    ax[1,1].set_xlabel('qwerty dimension 1')
    ax[1,1].set_ylabel('qwerty dimension 2')
    ax[1,1].legend()
    
    plt.tight_layout()
    plt.show()

# Now for the optimizer comparison

'''
#run the model for one optimizer
optimizerType = 'Adam'
trainAcc,testAcc,losses,net = function2trainTheModel(optimizerType)
'''

# average performance
performance = []

learningrates = np.logspace(-4, -1, num=20)
L2lambda = np.linspace(0, 0.1, 6)
#for opto in ['SGD','RMSprop','Adam']:
    
numepochs = 50
trainAccList = np.zeros((numepochs, len(L2lambda)))
testAccList = np.zeros((numepochs, len(L2lambda)))

for i,L2 in enumerate(L2lambda):
    trainAcc,testAcc,losses,net = function2trainTheModel('Adam', L2)

    trainAccList[:,i] = trainAcc
    testAccList[:,i] = testAcc
    
#plotTheResults(opto)
fig, ax = plt.subplots(1, 2, figsize = (12,6))
ax[0].plot(trainAccList)
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].set_title('Train Accuracy')
ax[0].legend(L2lambda)
ax[1].plot(testAccList)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Test Accuracy')
ax[1].legend(L2lambda)

plt.show()



