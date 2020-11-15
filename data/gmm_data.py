from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch


class GMMData(Dataset):
    """
    GMM dataset: X is sample from Gaussian i, Y is the id of the Gaussian
    Gaussian centers and variance are defined in config
    """
    gaussians = []

    def __init__(self, dp, mode='train'):
        self.dp = dp
        if dp.prior is None: 
            dp.prior = np.ones(dp.num_classes, dtype=float) / dp.num_classes

        torch.manual_seed(dp.seed)
        if not len(self.__class__.guassians):
            self.init_gauss()

    def init_gauss(self):
        self.__class__.gaussians = []
        for ind in range(self.dp.num_classes):
            mean = torch.randint(-100, 100, (self.dp.gauss_dim,))
            m = torch.rand((self.dp.gauss_dim, self.dp.gauss_dim))
            cov_mtx =  m@m.T + torch.eye(self.dp.gauss_dim)
            m = MultivariateNormal(mean.float(), cov_mtx)
            self.__class__.gaussians.append(m)

    def sample(self, nsample):
        """
        Sample the underlying distribution
        """
        xs = []
        for _ in range(nsample):
            rand_index = np.random.choice(np.arange(self.dp.num_classes), p=self.dp.prior)
            gaussian = self.__class__.gaussians[rand_index]
            xs.append(gaussian.sample())
        return xs

    def log_prob(self, x):
        """
        Compute likelihood of point x
        """
        probs = [g.log_prob(x) for g in self.__class__.gaussians]
        return torch.sum(probs) / self.dp.num_classes

    def __len__(self):
        return self.dp.num_samples

    def __getitem__(self, idx):
        xs, ys = [], []
        if type(idx) is int: idx = [idx]
        for _ in range(len(idx)):
            rand_index = np.random.choice(np.arange(self.dp.num_classes), p=self.dp.prior)
            gaussian = self.__class__.gaussians[rand_index]
            item = gaussian.sample()
            xs.append(item)
            ys.append(rand_idx)

        return xs, ys


class GMMTeacherData(GMMData):
    """
    Same structure as above, but uses teacher to label data
    """
    def __getitem__(self, idx):
        xs, ys = [], []
        if type(idx) is int: idx = [idx]
        for _ in range(len(idx)):
            rand_index = np.random.choice(np.arange(self.dp.num_classes), p=self.dp.prior)
            gaussian = self.__class__.gaussians[rand_index]
            item = gaussian.sample()
            y = self.dp.teacher(item)
            xs.append(item)
            ys.append(y)

        return xs, ys
