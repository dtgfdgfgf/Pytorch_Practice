import torch
class Config:
    def __init__(self):
        self.img_size = 64
        self.batch_size = 256
        self.latent_dim = 100
        self.num_epochs = 50
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.data_root = './CIFAR10'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")