import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

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
       
def loss_fun()        
        
        
        
        
        
        



