import torch
from torch.distributions import Categorical

def empirical_kl(distr_p, distr_q, nsamples=10**3):
    mean = torch.zeros(())
    n_batches = nsamples // 10**3
    for _ in range(n_batches):
        X = distr_p.sample(10**3)
        log_p = distr_p.log_prob(X)
        log_q = distr_q.log_prob(X)

        # Allow for 0.1% to have numerical error
        inf_locs = torch.isinf(log_q)
        n_inf = torch.sum(inf_locs)
        if n_inf > 0 and n_inf < 10:
            log_q = log_q[torch.where(~inf_locs)]
            log_p = log_p[torch.where(~inf_locs)]

        mean += torch.mean(log_p) - torch.mean(log_q)
    return mean / n_batches

def empirical_entropy(tensor):
    tensor = tensor.float()
    opts = torch.unique(tensor)
    entr = 0
    for opt in opts:
        p = torch.sum(tensor==opt) / float(len(tensor))
        p = p.float()
        if p > 0: entr -= p * torch.log2(p)
    return entr

def empirical_posterior_entropy(distr_p,  model, nsamples=10**3):
    ent = 0
    for ind in range(distr_p.num_classes):
        X = distr_p.sample(nsamples, class_index=ind)
        ys = model(X)
        ys = torch.argmax(ys, dim=1)
        ent += empirical_entropy(ys)
    return ent / distr_p.num_classes

def empirical_distr_entropy(distr_p, model, nsamples=10**3):
    X = distr_p.sample(nsamples)
    ys = model(X)
    ys = torch.argmax(ys, dim=1)
    return empirical_entropy(ys)
