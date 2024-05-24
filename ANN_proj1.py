"""
Predict the sum of two numbers.
"""

import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

from sklearn.model_selection import train_test_split

np.random.seed(0)
num = 1000
X = np.random.randint(-10, 11, (num,2))
y = np.sum(X,axis=1)

dataT = torch.tensor(X).float()
label = torch.tensor(y).float()
label = label[:,None]

train_data, test_data, train_labels, test_labels = train_test_split(dataT, label, test_size=0.1)

#print(X, y)

FFN = nn.Sequential(
    nn.Linear(2,16),
    nn.ReLU(),
    nn.Linear(16,16),
    nn.ReLU(),
    nn.Linear(16, 1)
    # It is a prediction of continuous values, do not add the activated function at the end.
    )

lossfun = nn.MSELoss()
optimizer = torch.optim.SGD(FFN.parameters(), lr=0.01)

times = 10
epoch = 1000
train_acc = torch.zeros(times)
test_acc = torch.zeros(times)
train_loss = torch.zeros(times, epoch)
test_loss = torch.zeros(times, epoch)

for time in range(times):
    for epochi in range(epoch):
        
        FFN.train()
        
        yHat = FFN(train_data)
        
        loss = lossfun(yHat, train_labels)
        train_loss[time][epochi] = loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        FFN.eval()
        with torch.no_grad():
            yHat = FFN(test_data)
        test_loss[time][epochi] = lossfun(yHat, test_labels)
        #print(loss)

for i in range(times): 
    train_acc[i] = torch.mean(train_loss[i]).detach()
    test_acc[i] = torch.mean(test_loss[i]).detach()
for i in range(times):
    print(f'train:{torch.mean(train_loss[i])}, test:{torch.mean(test_loss[i])}')
 
plt.plot(train_acc, label = 'train') 
plt.plot(test_acc, label = 'test')    
plt.legend()
plt.show()         
    
    
    
    
    
    










