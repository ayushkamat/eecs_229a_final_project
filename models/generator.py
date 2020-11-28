from .mlp import mlp
from torch import nn, normal
import torch

class generator(nn.Module):
    def __init__(self, mp):
        super().__init__()
        self.mp = mp
        self.network = mlp(mp)
        self.to(mp.device)
        self.perturb = torch.eye(self.mp.gauss_dim).to(self.mp.device)
    
    def forward(self, batch_size=1):
        noise = normal(self.mp.noise_mean, self.mp.noise_std, size=(self.mp.input_size,)).to(self.mp.device)
        output = self.network(noise).squeeze()
        gaussian_means = output[:self.mp.gauss_dim]
        gaussian_log_stds = output[self.mp.gauss_dim:].view(self.mp.gauss_dim, self.mp.gauss_dim)
        a = torch.exp(gaussian_log_stds)
        gaussian_stds =  a@a.T + self.perturb
        return gaussian_means, gaussian_stds