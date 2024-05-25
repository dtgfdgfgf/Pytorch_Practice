import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import logging
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

class Config:
    def __init__(self):
        self.input_dim = 784
        self.hidden_dim = [512,256,128]
        self.latent_dim = 20
        self.lr = 0.001
        self.batch_size = 32
        self.num_epochs = 10
        self.opt = 'Adam'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = './MNIST_data'
        

def create_VAE(input_dim, hidden_dim, latent_dim):
    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            
            # encoder
            self.encoder_layers = nn.ModuleDict()
            prev_dim = input_dim
            for i, h_dim in enumerate(hidden_dim):
                self.encoder_layers[f'en{i}'] = nn.Linear(prev_dim, h_dim)
                prev_dim = h_dim
            
            # VAE Reparameterization
            self.mean_layer = nn.Linear(hidden_dim[-1], latent_dim)
            self.logvar_layer = nn.Linear(hidden_dim[-1], latent_dim)
            
            # decoder
            self.decoder_layers = nn.ModuleDict()
            prev_dim = latent_dim
            for i, h_dim in enumerate(reversed(hidden_dim)):
                self.decoder_layers[f'de{i}'] = nn.Linear(prev_dim, h_dim)
                prev_dim = h_dim
            self.decoder_layers['output'] = nn.Linear(prev_dim, input_dim)
           
        def encode(self, x):
            for i in range(len(self.encoder_layers)):
                x = torch.relu(self.encoder_layers[f'en{i}'](x))
            mean = self.mean_layer(x)
            logvar = self.logvar_layer(x)
            return mean, logvar

        def reparameterize(self, mean, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        
        def decode(self, z):
            for i in range(len(self.decoder_layers)-1):
                z = torch.relu(self.decoder_layers[f'de{i}'](z))
            return torch.sigmoid(self.decoder_layers['output'](z))
        
        def forward(self, x):
            mean, logvar = self.encode(x)
            z = self.reparameterize(mean, logvar)
            return self.decode(z), mean, logvar
        
    return VAE()
       
def loss_fun(re_x, x, mean, logvar):
    BCE = nn.functional.binary_cross_entropy(re_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD

def train(VAE, config, train_loader):
    
    optifun = getattr(torch.optim, config.opt)
    optimizer = optifun(VAE.parameters(), lr=config.lr)
    
    train_loss = torch.zeros(config.num_epochs)
    for epoch in range(config.num_epochs):
        VAE.train()
        batch_loss = []
        for batch_idx, (data, _) in enumerate(train_loader): # AE不需要標籤
            data = data.view(-1, config.input_dim).to(config.device) #重塑形狀為[batch_size, input_dim]
            re_data, mean, logvar = VAE(data)
            loss = loss_fun(re_data, data, mean, logvar)
            batch_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                 
        train_loss[epoch] = np.mean(batch_loss) 
        print(f'epoch{epoch}')
        
    return train_loss

def plot_train_loss(loss):
    
    plt.plot(loss, label='train')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
def main():
    config = Config()
    
    transform = transforms.Compose([
        transforms.ToTensor()
        ])
    
    train_data_path = os.path.join(config.data_dir, 'MNIST', 'raw')
    if not os.path.exists(train_data_path):
        download = True
    else:
        download = False
        
    train_dataset = datasets.MNIST(config.data_dir, train=True, download=download, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    print(config.device)
    VAE = create_VAE(config.input_dim, config.hidden_dim, config.latent_dim).to(config.device)
    
    train_loss = train(VAE, config, train_loader)
    plot_train_loss(train_loss)

    
    

if __name__ == "__main__":
    main()





        
        
        
        
        
        
        



