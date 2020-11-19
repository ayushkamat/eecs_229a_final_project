from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch


class GMMData(Dataset):
    """
    GMM dataset: X is sample from Gaussian i, Y is the id of the Gaussian
    Gaussian centers and variance are defined in config
    """

    def __init__(self, dp, means=None, cov_mtx=None, mode='train'):
        self.dp = dp
        self.gaussians = []
        self.means = means
        self.cov_mtx = cov_mtx
        self.mode = mode
        self.init_gauss(means, cov_mtx)

    def init_gauss(self, means=None, cov_mtx=None):
        self.gaussians = []
        _means = []
        for ind in range(self.dp.num_classes):
            lb, ub = self.dp.loc_lower, self.dp.loc_upper
            mean = means[ind] if means is not None else torch.randint(lb, ub+1, (self.dp.gauss_dim,))
            _means.append(mean)
            if cov_mtx is None:
                m = torch.rand((self.dp.gauss_dim, self.dp.gauss_dim))
                cov_mtx =  m@m.T + torch.eye(self.dp.gauss_dim)
            m = MultivariateNormal(mean.float(), cov_mtx)
            self.gaussians.append(m)
        self.means = torch.tensor(_means).to(self.dp.device)

    def copy(self, std=0):
        means = self.means
        cov_mtx = self.cov_mtx
        if std > 0:
            means += torch.normal(torch.zeros(self.dp.gauss_dim), torch.tensor(std).to(self.dp.device))
            m = torch.rand((self.dp.gauss_dim, self.dp.gauss_dim))
            cov_mtx =  m@m.T + torch.eye(self.dp.gauss_dim)

        return GMMData(self.dp, means, cov_mtx, self.mode)

    def sample(self, nsample):
        """
        Sample the underlying distribution
        """
        xs = []
        for _ in range(nsample):
            rand_index = np.random.choice(np.arange(self.dp.num_classes))
            gaussian = self.gaussians[rand_index]
            xs.append(gaussian.sample())
        return torch.stack(xs).to(self.dp.device)

    def log_prob(self, x):
        """
        Compute likelihood of point x
        """
        if len(x.shape) == 1:
            probs = torch.stack([torch.exp(g.log_prob(x)) for g in self.gaussians]).to(self.dp.device)
            return torch.log(torch.sum(probs) / self.dp.num_classes)

        out = []
        for pt in x:
            probs = torch.stack([torch.exp(g.log_prob(pt)) for g in self.gaussians]).to(self.dp.device)
            out.append(torch.sum(probs) / self.dp.num_classes)
        return torch.log(torch.stack(out).to(self.dp.device))

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

