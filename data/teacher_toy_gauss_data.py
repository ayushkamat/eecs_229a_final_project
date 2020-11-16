from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch


class TeacherToyGauss(Dataset):
    """Teacher Toy dataset: X is a random input, Y is the output from the teacher applied to X"""

    def __init__(self, dp, mode='train'):
        self.dp = dp
        self.gen_data()

    def gen_data(self):
        self.xs = []
        self.ys = []
        with torch.no_grad():
            for _ in range(self.dp.num_samples):
                rand_input = torch.rand((1, self.dp.gauss_dim))
                y = self.dp.teacher(rand_input)
                self.xs.append(rand_input.squeeze())
                self.ys.append(y)

    def __len__(self):
        return self.dp.num_samples

    def __getitem__(self, idx):
        return self.xs[idx].to(self.dp.device), self.ys[idx].squeeze().to(self.dp.device)