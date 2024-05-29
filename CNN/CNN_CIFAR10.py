import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from config import Config
        
def createCNN(config):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=config.layer1['out_channels'], 
                               kernel_size=config.layer1['kernel_size'], 
                               stride=config.layer1['stride'], 
                               padding=config.layer1['padding'])
            
            self.conv2 = nn.Conv2d(in_channels=config.layer1['out_channels'], 
                                   out_channels=config.layer2['out_channels'], 
                                   kernel_size=config.layer2['kernel_size'], 
                                   stride=config.layer2['stride'], 
                                   padding=config.layer2['padding'])
            
            self.conv3 = nn.Conv2d(in_channels=config.layer2['out_channels'], 
                               out_channels=config.layer3['out_channels'], 
                               kernel_size=config.layer3['kernel_size'], 
                               stride=config.layer3['stride'], 
                               padding=config.layer3['padding'])
            
            self.dropout = nn.Dropout(config.dropout_rate)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1_input_dim = self.get_fc1_input_dim(config)
            self.fc1 = nn.Linear(self.fc1_input_dim, config.fc_layers['fc1'])  
            self.fc2 = nn.Linear(config.fc_layers['fc1'], config.fc_layers['fc2'])
            self.fc3 = nn.Linear(config.fc_layers['fc2'], config.fc_layers['output'])
            
        def get_fc1_input_dim(self, config):
            
            size = 32
            size = (size - config.layer1['kernel_size'] + 2 * config.layer1['padding']) // config.layer1['stride'] + 1
            size = (size - 2) // 2 + 1  # MaxPool
            size = (size - config.layer2['kernel_size'] + 2 * config.layer2['padding']) // config.layer2['stride'] + 1
            size = (size - 2) // 2 + 1  # MaxPool
            size = (size - config.layer3['kernel_size'] + 2 * config.layer3['padding']) // config.layer3['stride'] + 1
            size = (size - 2) // 2 + 1  # MaxPool
            return size * size * config.layer3['out_channels']
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, self.fc1_input_dim)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.fc3(x)
            return x
        
    net = CNN()
    
    return net

def train(CNNnet, config, train_loader, dev_loader, test_loader):
    
    CNNnet.to(config.device)
    lossfun = nn.CrossEntropyLoss()
    optifun = getattr(optim, config.opt)
    optimizer = optifun(CNNnet.parameters(), lr=config.lr)   
    
    train_loss = torch.zeros(config.epochs)
    train_acc = torch.zeros(config.epochs)
    dev_loss = torch.zeros(config.epochs)
    dev_acc = torch.zeros(config.epochs)
    for epoch in range (config.epochs):
        
        CNNnet.train()
        batch_loss = []
        batch_acc = []
        
        for X, y in train_loader:
            X, y = X.to(config.device), y.to(config.device)
            yHat = CNNnet(X)
            loss = lossfun(yHat, y)
            
            _, predicted = torch.max(yHat.data, 1)
            batch_correct = (predicted == y).sum().item() # 因為item所以不用.cpu
            batch_total = y.size(0)
            batch_loss.append(loss.item())
            batch_acc.append(100 * batch_correct / batch_total)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss[epoch] = np.mean(batch_loss)
        train_acc[epoch] = np.mean (batch_acc)
        print(f"Epoch {epoch+1}/nTraining Loss: {train_loss[epoch]}, Training Accuracy: {train_acc[epoch]}%")
        
        dev_loss[epoch], dev_acc[epoch] = evaluate_model(CNNnet, config, dev_loader, lossfun)
        print(f"dev Loss: {dev_loss[epoch]}, dev Accuracy: {dev_acc[epoch]}%")
    
    return train_loss, train_acc, dev_loss, dev_acc

def evaluate_model(net, config, dev_loader, lossfun):
    net.eval()
    net.to(config.device)

    total_loss = 0.0
    correct_num = 0
    label_num = 0
    
    with torch.no_grad():
        for X, y in dev_loader:
            X, y = X.to(config.device), y.to(config.device)
            yHat = net(X)
            loss = lossfun(yHat, y)
            total_loss += loss.item()
            _, predicted = torch.max(yHat.data, 1)
            correct_num += (predicted == y).sum().item()
            label_num += y.size(0)
    
    average_loss = total_loss / len(dev_loader)
    accuracy = 100 * correct_num / label_num
    return average_loss, accuracy

def test(net, test_loader, config):

    lossfun = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(net, config, test_loader, lossfun)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}%")
    
    return test_loss, test_acc

def plot_all(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc):
    
    fig,ax = plt.subplots(1,2,figsize=(16,5))
    
    ax[0].plot(train_loss, label='train')
    ax[0].plot(dev_loss, label='dev')
    #ax[0].plot(test_loss, label='test')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_label('Loss')
    ax[0].legend()
    
    ax[1].plot(train_acc, label='train')
    ax[1].plot(dev_acc, label='dev')
    #ax[1].plot(test_acc, label='test')
    ax[1].set_title('Acc')
    ax[1].set_xlabel('Epoch')
    ax[1].set_label('Acc(%)')
    ax[1].legend()
    
    plt.tight_layout(pad=2.0)
    plt.show()

    
    
def main():
    
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    train_val_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                                     download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                                download=True, transform=transform)
    
    batchSize = 32
    train_size = int(0.85 * len(train_val_dataset))
    dev_size = len(train_val_dataset) - train_size
    train_dataset, dev_dataset = random_split(train_val_dataset, [train_size, dev_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=batchSize, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False, num_workers=2)
    
    config = Config()
    CNNnet = createCNN(config)
    train_loss, train_acc, dev_loss, dev_acc = train(CNNnet, config, train_loader, dev_loader, test_loader)
    test_loss, test_acc = test(CNNnet, test_loader, config)
    
    plot_all(train_loss, train_acc, dev_loss, dev_acc, test_loss, test_acc)

if __name__ == "__main__":
    main()







