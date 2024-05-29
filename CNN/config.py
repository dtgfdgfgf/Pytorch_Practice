import torch

class Config:
    def __init__(self):
        self.layer1 = {
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        }
        self.layer2 = {
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        }
        self.layer3 = {
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1
        }
        self.fc_layers = {
        "fc1": 64,
        "fc2": 32,
        "output": 10  # Number of classes for CIFAR-10
        }
        
        self.epochs = 50
        self.opt = "Adam"
        self.lr = 0.001
        self.dropout_rate = 0.25
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')