import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from config import Config

# Create Generator
def createGen(config):
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            
            # convolution layers
            self.conv1 = nn.ConvTranspose2d(config.latent_dim,512, 4, 1, 0, bias=False)
            self.conv2 = nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False)
            self.conv3 = nn.ConvTranspose2d(256,128, 4, 2, 1, bias=False)
            self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
            self.conv5 = nn.ConvTranspose2d(64,   3, 4, 2, 1, bias=False)
            
            # batchnorm
            self.bn1 = nn.BatchNorm2d(512)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d( 64)

        # Apply layers with ReLU and Tanh activations
        def forward(self,x):
            x = F.relu( self.bn1(self.conv1(x)) )
            x = F.relu( self.bn2(self.conv2(x)) )
            x = F.relu( self.bn3(self.conv3(x)) )
            x = F.relu( self.bn4(self.conv4(x)) )
            x = torch.tanh( self.conv5(x) )
            return x
        
    net = Generator()
    return net

# Create Discriminator
def createDis(config):
    class Discrimator(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Convolution layers
            self.conv1 = nn.Conv2d(  3, 64, 4, 2, 1, bias=False)
            self.conv2 = nn.Conv2d( 64,128, 4, 2, 1, bias=False)
            self.conv3 = nn.Conv2d(128,256, 4, 2, 1, bias=False)
            self.conv4 = nn.Conv2d(256,512, 4, 2, 1, bias=False)
            self.conv5 = nn.Conv2d(512,  1, 4, 1, 0, bias=False)
            
            # batchnorm
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(512)
            
        # Apply layers with ReLU and Sigmoid activations
        def forward(self,x):
            x = F.relu( self.conv1(x))
            x = F.relu( self.conv2(x))
            x = self.bn2(x)
            x = F.relu( self.conv3(x))
            x = self.bn3(x)
            x = F.relu( self.conv4(x))
            x = self.bn4(x)
            return torch.sigmoid(self.conv5(x)).view(-1,1)
        
    net = Discrimator()
    return net

# Training function
def train(config, data_loader, dnet, gnet):
    
    lossfun = nn.BCELoss()
    d_optimizer = optim.Adam(dnet.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    g_optimizer = optim.Adam(gnet.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        
    epoch_losses = []
    epoch_disDecs = []
    
    for epochi in range(config.num_epochs):
        print(f'epoch{epochi} started')
        batch_losses = []
        batch_disDecs = []
        for data,_ in data_loader:
            data = data.to(config.device)
            real_labels = torch.ones(config.batch_size,1).to(config.device)
            fake_labels = torch.zeros(config.batch_size,1).to(config.device)
            
            # Train the discriminator
            d_real = dnet(data)
            d_real_loss = lossfun(d_real, real_labels)
            
            fake_data   = torch.randn(config.batch_size,100,1,1).to(config.device) # random numbers to seed the generator
            fake_images = gnet(fake_data)
            d_fake = dnet(fake_images)
            d_fake_loss = lossfun(d_fake, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # Train the generator
            fake_images = gnet(torch.randn(config.batch_size,100,1,1).to(config.device))
            d_fake   = dnet(fake_images)
            
            g_loss = lossfun(d_fake, real_labels)
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            d1 = torch.mean((d_real>0.5).float()).detach()
            d2 = torch.mean((d_fake>0.5).float()).detach()
            batch_disDecs.append([d1.item(), d2.item()])  
            batch_losses.append([d_loss.item(), g_loss.item()])
        
        epoch_disDecs.append(np.mean(batch_disDecs, axis=0))
        epoch_losses.append(np.mean(batch_losses, axis=0))
        
        print(f'epoch{epochi} finished')
    
    return epoch_losses, epoch_disDecs

def plot_losses(losses):
    losses = np.array(losses)
    plt.figure(figsize=(10, 5))
    plt.plot(losses[:, 0], label='Discriminator Loss')
    plt.plot(losses[:, 1], label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Losses Over Time')
    plt.show()

def plot_discriminator_decisions(decisions):
    decisions = np.array(decisions)
    plt.figure(figsize=(10, 5))
    plt.plot(decisions[:, 0], label='Real Image Confidence')
    plt.plot(decisions[:, 1], label='Fake Image Confidence')
    plt.xlabel('Epoch')
    plt.ylabel('Confidence')
    plt.legend()
    plt.title('Discriminator Decisions Over Time')
    plt.show()
    
def main():
    
    transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    config = Config()
    trainset = torchvision.datasets.CIFAR10(root=config.data_root, train=True, download=True, transform=transform)
    subset_indices = torch.arange(5000)  # Use the first 5000 images
    subset = torch.utils.data.Subset(trainset, subset_indices)

    train_loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True,drop_last=True, num_workers=4)
    
    G_net = createGen(config).to(config.device)
    D_net = createDis(config).to(config.device)
    
    epoch_losses, epoch_disDecs = train(config, train_loader, D_net, G_net)
    plot_losses(epoch_losses)
    plot_discriminator_decisions(epoch_disDecs)
    
if __name__ == "__main__":
    main()




















