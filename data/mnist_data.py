from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.distributions as D

class MNISTData(MNIST):
    """Mnist Dataset wrapper"""

    def __init__(self, dp, mode='train'):
        self.dp = dp
        resolution = dp.resolution or (28, 28)
        directory = dp.dir or './cache/mnist'
        classes = dp.classes

        super().__init__(directory,
            train=(mode == 'train'),
            download=True,
            transform=transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ) 

        # filter classes and flatten data
        # classes are reindexed using dp.classes as a map
        self.clean_data(classes)

        self.input_prior = D.MultivariateNormal(torch.zeros(resolution[0]*resolution[1],).to(self.dp.device), torch.eye(resolution[0]*resolution[1],).to(self.dp.device))
        
    def clean_data(self, classes):
        if classes and len(classes) < 10:
            # filters data to classes
            indices = torch.as_tensor([digit in classes for digit in self.targets])
            x, y = self.data[indices], self.targets[indices]
            # reindex labels
            new_y = torch.zeros_like(y)
            for index, digit in enumerate(classes):
                new_y[y == digit] = index
            self.data, self.targets = x, new_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        # flatten and normalize data
        x_flattened = x.flatten()
        return x_flattened.to(self.dp.device), torch.tensor(y).to(self.dp.device)