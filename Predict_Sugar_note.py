"""
By comparing the correlation between model predictions and true values, 
the accuracy and generalization ability of the model can be further verified.
"""
# for DL modeling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

# for number-crunching
import numpy as np
import scipy.stats as stats

# for dataset management
import pandas as pd

# for timing computations
import time

# for data visualization
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

# import the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url,sep=';')
data = data[data['total sulfur dioxide']<200] # drop a few outliers

# 取label
labels = torch.tensor( data['residual sugar'].values ).float()
labels = labels[:,None] # transform to matrix

# z-score all columns except for quality
cols2zscore = data.keys()
cols2zscore = cols2zscore.drop(['quality'])
cols2zscore = cols2zscore.drop(['residual sugar'])
data[cols2zscore] = data[cols2zscore].apply(stats.zscore)

# convert from pandas dataframe to tensor
dataT  = torch.tensor( data[cols2zscore].values ).float()

# split the data
train_data,test_data, train_labels,test_labels = train_test_split(dataT, labels, test_size=0.1)

# convert into PyTorch Datasets
train_dataDataset = TensorDataset(train_data,train_labels)
test_dataDataset  = TensorDataset(test_data,test_labels)

#print(data.keys())

# create a class for the model

class ANNwine(nn.Module):
    def __init__(self):
        super().__init__()
      
        ### input layer
        self.input = nn.Linear(10,16)
      
        ### hidden layers
        self.fc1 = nn.Linear(16,32)
        self.fc2 = nn.Linear(32,32)
      
        ### output layer
        self.output = nn.Linear(32,1)
    
    # forward pass
    def forward(self,x):
        x = F.relu( self.input(x) )
        x = F.relu( self.fc1(x) ) # fully connected
        x = F.relu( self.fc2(x) )
        return self.output(x)

# global parameter
numepochs = 1000

# train the model
def trainTheModel():

    # loss function and optimizer
    lossfun = nn.MSELoss()
    optimizer = torch.optim.SGD(winenet.parameters(),lr=0.01)
    
    # initialize losses
    trainloss = torch.zeros(numepochs)
    testloss  = torch.zeros(numepochs)
    
    # loop over epochs
    for epochi in range(numepochs):
    
        # switch on training mode
        winenet.train()
      
        # loop over batches
        batchLoss = []
        for X,y in train_loader: #從train_loader中迭代取出每一批數據X跟對應的標籤Y
      
            # forward pass and loss
            yHat = winenet(X) # 向前傳播，計算預測輸出yHat，通過模型定義的 forward 方法自動完成的。
            loss = lossfun(yHat,y) #使用損失函數，計算loss
        
            # backprop
            optimizer.zero_grad() # 歸零所有累積的梯度，這是每次迭代開始前必需的步驟，以避免梯度在多次反向傳播中累積。
            loss.backward() # 觸發損失關於模型參數的梯度計算，這是訓練神經網路中的關鍵步驟。
            optimizer.step() # 根據計算得到的梯度更新模型的參數，這是實現模型學習的步驟。
        
            # loss from this batch
            batchLoss.append(loss.item()) #loss是一個張量，通過.item()將其轉換為Python浮點數，並將其添加到 batchLoss 列表中。這個列表用於存儲本訓練周期內所有批次的損失。
            
        # end of batch loop...

        trainloss[epochi] = np.mean(batchLoss) #在完成一個訓練周期的所有批次後，計算這個周期所有批次損失的平均值，並將這個平均值存儲在 trainloss 數組中的相應位置。這樣做可以追蹤模型在每個訓練周期的平均表現，有助於監控訓練過程中的損失變化，從而評估模型的訓練效果是否在逐漸改善。
  
        # test accuracy
        winenet.eval() #進入評估模式
        X,y = next(iter(test_loader)) # 從test dataloader 提取第一個batch(通常也可能是唯一的)
        with torch.no_grad(): #告訴PyTorch在這個代碼塊內部不追蹤用於自動梯度計算的操作，可以減少內存消耗並加速計算。在這種模式下，模型不會進行反向傳播或任何與訓練相關的更新。適用在test環節
            yHat = winenet(X) #計算預測值
        testloss[epochi] = lossfun(yHat,y).item() #計算預測值與真實值之間的loss
    
    # function output
    return trainloss, testloss

# create dataloader object
batchsizes = 32
train_loader = DataLoader(train_dataDataset,batch_size=batchsizes, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataDataset,batch_size=test_dataDataset.tensors[0].shape[0])
 
# create and train a model
winenet = ANNwine()
trainloss,testloss = trainTheModel()

# plot some results
fig,ax = plt.subplots(1,2,figsize=(17,7))

ax[0].plot(trainloss,label='Train loss')
ax[0].plot(testloss,label='Test loss')
ax[0].set_title('Losses')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('loss')
ax[0].legend()
ax[0].grid()

yHatTrain = winenet(train_data)
yHatTest  = winenet(test_data)

ax[1].plot(yHatTrain.detach(),train_labels,'ro')
ax[1].plot(yHatTest.detach(),test_labels,'b^')
ax[1].set_xlabel('Model-predicted sugar')
ax[1].set_ylabel('True sugar')
ax[1].set_title('Model predictions vs. observations')

# correlations between predictions and outputs
corrTrain = np.corrcoef(yHatTrain.detach().T,train_labels.T)[1,0]
corrTest  = np.corrcoef(yHatTest.detach().T, test_labels.T)[1,0]

ax[1].legend([ f'Train r={corrTrain:.3f}',f'Test r={corrTest:.3f}' ])




plt.show()







