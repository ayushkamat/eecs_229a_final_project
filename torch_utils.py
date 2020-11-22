import torch
from torch.distributions import Categorical

def empirical_kl(distr_p, distr_q, nsamples=10**3):
    mean = torch.zeros(())
    n_batches = nsamples // 10**3
    for _ in range(n_batches):
        X = distr_p.sample(10**3)
        log_p = distr_p.log_prob(X)
        log_q = distr_q.log_prob(X)

        mean += torch.mean(log_p) - torch.mean(log_q)
    return mean / n_batches

def empirical_entropy(tensor):
    return Categorical(probs=tensor).entropy()

def empirical_posterior_entropy(distr_p,  model, nsamples=10**3):
    ent = 0
    for ind in range(distr_p.num_classes):
        X = distr_p.sample(nsamples, class_index=ind)
        ys = model(X)
        ys = torch.argmax(ys, dim=1)
        ent += empirical_entropy(ys)
    return ent / distr_p.num_classes
        
