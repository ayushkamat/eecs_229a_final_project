from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import torch


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
        self.xs, self.ys = self.clean_data(classes)

        # clean variables
        self.data, self.targets = None, None
        
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

        # flatten and normalize data
        flattened = self.data.view(self.data.shape[0], -1)
        return list(flattened.float() / 255), list(self.targets)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx].float().to(self.dp.device), self.ys[idx].squeeze().to(self.dp.device)
