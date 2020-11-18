from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch


class GMMData(Dataset):
    """
    GMM dataset: X is sample from Gaussian i, Y is the id of the Gaussian
    Gaussian centers and variance are defined in config
    """

    def __init__(self, dp, mode='train'):
        self.dp = dp
        self.gaussians = []

        torch.manual_seed(dp.seed)
        self.init_gauss()

    def init_gauss(self):
        self.gaussians = []
        for ind in range(self.dp.num_classes):
            mean = torch.randint(-100, 100, (self.dp.gauss_dim,))
            m = torch.rand((self.dp.gauss_dim, self.dp.gauss_dim))
            cov_mtx =  m@m.T + torch.eye(self.dp.gauss_dim)
            m = MultivariateNormal(mean.float(), cov_mtx)
            self.gaussians.append(m)

    def sample(self, nsample):
        """
        Sample the underlying distribution
        """
        xs = []
        for _ in range(nsample):
            rand_index = np.random.choice(np.arange(self.dp.num_classes))
            gaussian = self.gaussians[rand_index]
            xs.append(gaussian.sample())
        return xs

    def log_prob(self, x):
        """
        Compute likelihood of point x
        """
        probs = [g.log_prob(x) for g in self.gaussians]
        return torch.sum(probs) / self.dp.num_classes

    def __len__(self):
        return self.dp.num_samples

    def __getitem__(self, idx):
        if type(idx) is int:
            rand_index = np.random.choice(np.arange(self.dp.num_classes))
            gaussian = self.gaussians[rand_index]
            item = gaussian.sample()
            return item.to(self.dp.device), torch.tensor(rand_index).to(self.dp.device)

        xs, ys = [], []
        for _ in range(len(idx)):
            rand_index = np.random.choice(np.arange(self.dp.num_classes))
            gaussian = self.gaussians[rand_index]
            item = gaussian.sample()
            xs.append(item)
            ys.append(torch.tensor(rand_index))

        return torch.stack(xs).to(self.dp.device), torch.stack(ys).to(self.dp.device)


class GMMTeacherData(GMMData):
    """
    Same structure as above, but uses teacher to label data
    """
    def __getitem__(self, idx):
        if type(idx) is int:
            rand_index = np.random.choice(np.arange(self.dp.num_classes))
            gaussian = self.gaussians[rand_index]
            item = gaussian.sample()
            y = self.dp.teacher(item)
            return item.to(self.dp.device), y.to(self.dp.device)

        xs, ys = [], []
        for _ in range(len(idx)):
            rand_index = np.random.choice(np.arange(self.dp.num_classes))
            gaussian = self.gaussians[rand_index]
            item = gaussian.sample()
            y = self.dp.teacher(item)
            xs.append(item)
            ys.append(y)

        return torch.stack(xs).to(self.dp.device), torch.stack(ys).to(self.dp.device)

