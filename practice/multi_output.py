import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# create data

nPerClust = 100
blur = 1

A = [  1, 5 ]
B = [  3, 1 ]
C = [  5, 5 ]

# generate data 3 groups
a = [ A[0]+np.random.randn(nPerClust)*blur , A[1]+np.random.randn(nPerClust)*blur ]
b = [ B[0]+np.random.randn(nPerClust)*blur , B[1]+np.random.randn(nPerClust)*blur ]
c = [ C[0]+np.random.randn(nPerClust)*blur , C[1]+np.random.randn(nPerClust)*blur ]

# true labels
#labels_np = np.vstack((np.zeros((nPerClust,1)),np.ones((nPerClust,1)), np.full((100,1), 2)))
labels_np = np.concatenate([np.zeros(nPerClust), np.ones(nPerClust), np.full(nPerClust, 2)])

# concatanate into a matrix
data_np = np.hstack((a,b,c)).T

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).long()

# model architecture
ANNiris = nn.Sequential(
    nn.Linear(2,64),   # input layer
    nn.ReLU(),         # activation
    nn.Linear(64,32),  # hidden layer
    nn.ReLU(),      
    nn.Linear(32,64),  # hidden layer
    nn.ReLU(),         # activation# activation
    nn.Linear(64,3),   # output layer
      )

# loss function
lossfun = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(ANNiris.parameters(),lr=0.01)

numepochs = 1000

# initialize losses
losses = torch.zeros(numepochs)
ongoingAcc = []

# loop over epochs
for epochi in range(numepochs):

    # forward
    yHat = ANNiris(data)
    
    # compute loss
    loss = lossfun(yHat,labels)
    losses[epochi] = loss
    
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # compute accuracy
    matches = torch.argmax(yHat,axis=1) == labels # booleans (false/true)
    matchesNumeric = matches.float()              # convert to numbers (0/1)
    accuracyPct = 100*torch.mean(matchesNumeric)  # average and x100
    ongoingAcc.append( accuracyPct )              # add to list of accuracies
    
# final forward pass
predictions = ANNiris(data)

predlabels = torch.argmax(predictions,axis=1)
totalacc = 100*torch.mean((predlabels == labels).float())

# report accuracy
print('Final accuracy: %g%%' %totalacc)

# plot
fig,ax = plt.subplots(1,2,figsize=(13,4))

ax[0].plot(losses.detach())
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('epoch')
ax[0].set_title('Losses')

ax[1].plot(ongoingAcc)
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].set_title('Accuracy')
plt.show()

sm = nn.Softmax(1)
fig = plt.figure(figsize=(10,4))

plt.plot(sm(yHat.detach()),'s-',markerfacecolor='w')
plt.xlabel('Stimulus number')
plt.ylabel('Probability')
plt.legend(['0','1','2'])
plt.show()



