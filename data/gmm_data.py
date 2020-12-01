from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal

import copy
import numpy as np
import torch


class GMMData(Dataset):
    """
    GMM dataset: X is sample from Gaussian i, Y is the id of the Gaussian
    Gaussian centers and variance are defined in config
    """

    def __init__(self, dp, means=None, cov_mtxs=None, mode='train'):
        torch.set_default_dtype(torch.float64)
        self.dp = dp
        self.gaussians = []
        self.num_classes = self.dp.num_classes
        self.means = means
        self.cov_mtxs = cov_mtxs
        self.mode = mode
        self.init_gauss(means, cov_mtxs)

    def init_gauss(self, means=None, cov_mtxs=None):
        self.gaussians = []
        _means = []
        _cov_mtx = []
        for ind in range(self.dp.num_classes):
            lb, ub = self.dp.loc_lower, self.dp.loc_upper
            mean = means[ind] if means is not None else torch.randint(lb, ub+1, (self.dp.gauss_dim,))
            _means.append(mean)
            if cov_mtxs is None:
                m = torch.rand((self.dp.gauss_dim, self.dp.gauss_dim))
                cov_mtx = m@m.T + 1e-1*torch.eye(self.dp.gauss_dim)
            else:
                cov_mtx = cov_mtxs[ind]
            _cov_mtx.append(cov_mtx)
            m = MultivariateNormal(mean.double(), cov_mtx)
            self.gaussians.append(m)
        self.means = torch.stack(_means).to(self.dp.device)
        self.cov_mtxs = torch.stack(_cov_mtx).to(self.dp.device)

    def copy(self, std=0, rescale=True):
        means = copy.copy(self.means)
        cov_mtxs = copy.copy(self.cov_mtxs)
        if std > 0:
            for ind in range(len(means)):
                means[ind] = means[ind] + torch.normal(torch.zeros(self.dp.gauss_dim), torch.tensor(std).to(self.dp.device))
                
        if rescale:
            for ind in range(len(means)):
                m = torch.rand((self.dp.gauss_dim, self.dp.gauss_dim))
                cov_mtxs[ind] = m@m.T + 1e-1*torch.eye(self.dp.gauss_dim)

        return GMMData(self.dp, means, cov_mtxs, self.mode)

    def sample(self, nsample=1, class_index=-1):
        """
        Sample the underlying distribution
        """
        xs = []
        for _ in range(nsample):
            idx = class_index if class_index >= 0 else np.random.choice(np.arange(self.dp.num_classes))
            gaussian = self.gaussians[idx]
            xs.append(gaussian.sample())
        if nsample == 1: return xs[0]
        return torch.stack(xs).to(self.dp.device)

    def log_prob(self, x):
        """
        Compute likelihood of point x
        """
        xs = x if len(x.shape) == 2 else [x]
        probs = []
        for g in self.gaussians:
            probs.append(torch.exp(g.log_prob(x).double()))
        probs = torch.stack(probs, axis=1)
        probs = torch.sum(probs, axis=1) / self.dp.num_classes
        probs = torch.log(probs).float()
        if len(x.shape) == 1: probs = probs[0]
        return probs

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

