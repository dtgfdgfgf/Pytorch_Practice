"""
Find the optimal model configuration by tuning multiple hyperparameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')

from sklearn.model_selection import train_test_split

import kaggle
import pandas as pd
import scipy.stats as stats

class Config:
    def __init__(self):
        self.nlayers = 2
        self.nunits = 300
        self.bn = True
        self.lr = np.linspace(0.0002, 0.002, 10)
        self.batch_size = 32
        self.num_epochs = 500
        self.opt = 'SGD'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout_rate = 0.1
        self.L2 = 0
      
def createFFN(nlayer, nunit, BN, dropout_rate):
    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()
            self.bnorms = nn.ModuleDict()
            self.dropouts = nn.ModuleDict()
            self.layers['input'] = nn.Linear(10, nunit)
            for i in range(nlayer):
                if BN:
                    self.bnorms[f'bnorm{i}'] = nn.BatchNorm1d(nunit)                
                self.layers[f'hidden{i}'] = nn.Linear(nunit, nunit)
                self.dropouts[f'dropout{i}'] = nn.Dropout(dropout_rate)
            self.layers['output'] = nn.Linear(nunit, 1)
        
        def forward(self, x):
            x = F.relu(self.layers['input'](x))
            for i in range(nlayer):
                if BN:
                    x = self.bnorms[f'bnorm{i}'](x)
                x = self.layers[f'hidden{i}'](x)
                x = F.relu(x)
                x = self.dropouts[f'dropout{i}'](x)
            x = self.layers['output'](x)
            #x = torch.sigmoid(x)
            return x
    return FFN()

def train(FFNnet, config, LR, train_loader, test_loader):
    
    numepoch = 500
    #lossfun = nn.CrossEntropyLoss()
    lossfun = nn.BCEWithLogitsLoss()
    optifun = getattr(torch.optim, config.opt)
    optimizer = optifun(FFNnet.parameters(), lr=LR)
    
    train_loss = torch.zeros(numepoch)
    #test_loss = torch.zeros(numepoch)
    
    trainAcc = np.zeros(numepoch)
    testAcc = np.zeros(numepoch)
    
    for epochi in range(numepoch):
        FFNnet.train()
        batch_loss = []
        batchAcc = []
        
        for X, y in train_loader:
            #print(y)
            yHat = FFNnet(X)
            loss = lossfun(yHat, y)
            #train_prob = F.softmax(yHat)
            #print(prob)
            batch_loss.append(loss.item())
            
            predictions = (torch.sigmoid(yHat)>0.5).float()
            batchAcc.append(100*torch.mean((predictions==y).float()))
            
            '''
            matches = torch.argmax(yHat, axis=1) == y
            matchesNumeric = matches.float()
            accuracyPct = 100*torch.mean(matchesNumeric)
            batchAcc.append(accuracyPct)
            '''
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss[epochi] = np.mean(batch_loss)        
        trainAcc[epochi] = np.mean(batchAcc)
        
        #print(train_loss[epochi])
        
        FFNnet.eval()
        X, y = next(iter(test_loader))
        with torch.no_grad():
            yHat = FFNnet(X)
        #testAcc[epochi] = 100*torch.mean((torch.argmax(yHat,axis=1) == y).float()) 
        predictions = (torch.sigmoid(yHat)>.5).float()
        testAcc[epochi] = 100*torch.mean((predictions==y).float())
        
        #print(f'epoch {epochi}')
    return trainAcc, testAcc, train_loss


def main():
                        
    #kaggle.api.dataset_download_files('redwankarimsony/heart-disease-data', path='C:/Users/USER/Desktop/pytorch_class/G_D/kaggle_dataset', unzip=True)
    
    #path='C:/Users/USER/Desktop/pytorch_class/G_D/kaggle_dataset/heart_disease_uci.csv'
    #data = pd.read_csv(path)
    config = Config()
    
    url  = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    data = pd.read_csv(url,sep=',',header=None)
    
    #print(data.keys())
    data.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','DISEASE']
    
    key = data.keys()
    key = key.drop(['sex','fbs','exang','DISEASE'])
    data = data.replace('?',np.nan).dropna()
    
    data['DISEASE'][data['DISEASE']>0] = 1
    label = data['DISEASE']
    
    for k in key:
        data[k] = pd.to_numeric(data[k])
    
    data = data.apply(stats.zscore)
    
    dataT = torch.tensor(data[key].values).float()
    labelT = torch.tensor(label.values).float()
    labelT = labelT[:,None]
    #print(labelT)
    
    train_data, test_data, train_labels, test_labels = train_test_split(dataT, labelT, test_size=0.1)
    
    train_data = TensorDataset(train_data, train_labels)
    test_data = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])
    
    
    fig, axes = plt.subplots(len(config.lr),1 , figsize=(15,15), sharex=True, sharey=True)
    fig.suptitle('Test Accuracy by Configuration')
    
    
    for i, LR in enumerate(config.lr):
        net = createFFN(config.nlayers, config.nunits, config.bn, config.dropout_rate)
        trainAcc, testAcc, train_loss = train(net, config, LR, train_loader, test_loader)
    
        ax = axes[i]
        ax.plot(trainAcc, label=f"train_LR={LR}")
        ax.plot(testAcc, label=f"test_LR={LR}")
        ax.set_title(f"Layers={config.nlayers}, Units={config.nunits}")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        print(f'train:{np.mean(trainAcc)}')
        print(f'test:{np.mean(testAcc)}')
    
    plt.tight_layout(pad=2.0)
    plt.show()

if __name__ == "__main__":
    main()










