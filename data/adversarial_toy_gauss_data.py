from torch.utils.data import Dataset
import numpy as np
import torch


class AdversarialToyGauss(Dataset):
    """Adversarial Toy dataset: X is a generated input, Y is the output from the teacher applied to X"""

    def __init__(self, dp, mode='train'):
        self.dp = dp
        torch.manual_seed(dp.seed)

    def __len__(self):
        return self.dp.num_samples

    def __getitem__(self, idx):
        gen_mean, gen_std = self.dp.generator()
        return gen_mean, gen_std