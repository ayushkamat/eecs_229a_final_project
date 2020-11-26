from .mlp import mlp
from torch import nn, normal

class GenerativeGenerator(nn.Module):
    def __init__(self, mp):
        super().__init__()
        self.mp = mp
        self.mean_network = mlp(mp)
        self.sigma_network = mlp(mp)
        self.to(mp.device)
    
    def forward(self, batch):
        means = self.mean_network(batch)
        sigmas = self.sigma_network(batch)
        return means, sigmas