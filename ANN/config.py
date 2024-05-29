import numpy as np
import torch

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