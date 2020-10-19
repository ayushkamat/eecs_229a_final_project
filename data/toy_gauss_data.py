from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch


class ToyGauss(Dataset):
    """Toy dataset: X is sample from Gaussian i, Y is the id of the Gaussian"""
    gaussians = [] # must be same between test and train sets

    def __init__(self, dp, mode='train'):
        self.dp = dp
        torch.manual_seed(dp.seed)
        if not self.gaussians:
            self.init_gauss()
        self.gen_data()

    def init_gauss(self):
        self.__class__.gaussians = []
        for _ in range(self.dp.num_classes):
            mean = torch.randint(100, (self.dp.gauss_dim,))
            m = torch.rand((self.dp.gauss_dim, self.dp.gauss_dim))
            cov_mtx =  m@m.T + torch.eye(self.dp.gauss_dim)
            m = MultivariateNormal(mean.float(), cov_mtx)
            self.__class__.gaussians.append(m)

    def gen_data(self):
        self.xs = []
        self.ys = []

        for _ in range(self.dp.num_samples):
            rand_index = torch.randint(self.dp.num_classes, (1, 1))
            gaussian = self.gaussians[rand_index]
            item = gaussian.sample()
            self.xs.append(item)
            self.ys.append(rand_index)

    def __len__(self):
        return self.dp.num_samples

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx].squeeze()