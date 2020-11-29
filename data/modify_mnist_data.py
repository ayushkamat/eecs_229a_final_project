from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import torch


class MNISTData(MNIST):
    """Mnist Dataset wrapper"""

    def __init__(self, dp, classes=None, noise_rate=0, corrupt_rate=0, mode='train'):
        self.dp = dp

        resolution = dp.resolution or (28, 28)
        directory = dp.dir or './cache/mnist'
        self.classes = list(range(10)) if classes is None else classes
        self.noise_rate = noise_rate
        self.corrupt_rate = corrupt_rate
        self.num_classes = len(classes)

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
        self.xs, self.ys = self.clean_data()

        # clean variables
        self.data, self.targets = None, None
        
    def clean_data(self):
        if len(self.classes) < 10:
            # filters data to classes
            indices = torch.as_tensor([digit.item() in self.classes for digit in self.targets])
            x, y = self.data[indices], self.targets[indices]
            # reindex labels
            new_y = torch.zeros_like(y)
            for index, digit in enumerate(self.classes):
                new_y[y == digit] = index
            self.data, self.targets = x, new_y

        xs = xs.to(self.dp.device)
        ys = self.targets.to(self.dp.device)
        xs = self.add_noise(xs)
        ys = self.corrupt_labels(ys)
        xs = ((self.data.double() / 255.) - 0.1307) / 0.3081
        xs = xs.reshape((-1, 1, 28, 28)).to(self.dp.device)
        return xs, ys

    def sample(self, size, class_index=None):
        xs = []
        for _ in range(size):
            inds = range(len(self.xs))
            if class_index is not None:
                assert class_index in self.ys
                inds = torch.where(self.ys == class_index)[0].cpu()
            ind = np.random.choice(inds)
            xs.append(self.xs[ind])
        return torch.stack(xs)

    def add_noise(self, xs):
        if self.noise_rate > 0:
            mask = (torch.rand(xs.shape) >= self.noise_rate).to(self.dp.device)
            noise = 255 * (torch.rand(xs.shape) > 0.5).to(self.dp.device)
            xs = mask * xs + (~mask) * noise
        return xs

    def corrupt_labels(self, ys):
        if self.corrupt_rate > 0:
            mask = (torch.rand(ys.shape) >= self.corrupt_rate).to(self.dp.device)
            corruption = torch.randint(0, len(self.classes), ys.shape).to(self.dp.device)
            ys = mask * ys + (~mask) * corruption
        return ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs, ys = self.xs[idx], self.ys[idx].squeeze()
        return xs, ys

