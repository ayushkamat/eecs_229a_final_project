from .mlp import mlp
from torch import nn, normal

class generator(nn.Module):
    def __init__(self, mp):
        super().__init__()
        self.mp = mp
        self.network = mlp(mp)
        self.to(mp.device)
    
    def forward(self, batch_size=1):
        noise = normal(self.mp.noise_mean, self.mp.noise_std, size=(batch_size, self.mp.input_size))
        generated = self.network(noise)
        return generated