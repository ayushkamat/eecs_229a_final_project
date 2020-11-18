import torch

def empirical_kl(distr_p, distr_q, nsamples=10**3):
    X = distr_p.sample(nsamples)
    log_p = distr_p.log_prob(X)
    log_q = distr_q.log_prob(X)
    return torch.mean(log_p) - torch.mean(log_q)
