import os
import math
import torch
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def bernoulli_log_pdf(x, probs):
    r"""Log-likelihood of data given ~Bernoulli(mu)
    @param x: PyTorch.Tensor
              ground truth input
    @param mu: PyTorch.Tensor
               Bernoulli distribution parameters
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    return torch.distributions.bernoulli.Bernoulli(probs=probs).log_prob(x)


def masked_bernoulli_log_pdf(x, mask, probs):
    dist = torch.distributions.bernoulli.Bernoulli(probs=probs)
    log_prob = dist.log_prob(x)
    return log_prob * mask.float()


def masked_gaussian_log_pdf(x, mask, mu, logvar):
    sigma = torch.exp(0.5 * logvar)
    dist = torch.distributions.normal.Normal(mu, sigma)
    log_prob = dist.log_prob(x)
    return log_prob * mask.float()


def normal_log_pdf(x, mu, logvar):
    scale = torch.exp(0.5 * logvar)
    return torch.distributions.normal.Normal(mu, scale).log_prob(x)


def standard_normal_log_pdf(x):
    mu = torch.zeros_like(x)
    scale = torch.ones_like(x)
    return torch.distributions.normal.Normal(mu, scale).log_prob(x)


def log_mean_exp(x, dim=1):
    """log(1/k * sum(exp(x))): this normalizes x.
    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    # m = torch.max(x, dim=dim, keepdim=True)[0]
    # return m + torch.log(torch.mean(torch.exp(x - m),
                        #  dim=dim, keepdim=True))
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[1])


def kl_divergence_standard_normal_prior(z_mu, z_logvar):
    kl_div = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_div = torch.sum(kl_div, dim=1)
    return kl_div


def kl_divergence_normal_prior(q_z_mu, q_z_logvar, p_z_mu, p_z_logvar):
    q = dist.Normal(q_z_mu, torch.exp(0.5 * q_z_logvar))
    p = dist.Normal(p_z_mu, torch.exp(0.5 * p_z_logvar))
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


def gentr_fn(alist):
    while 1:
        for j in alist:
            yield j


def product_of_experts(mu, logvar, eps=1e-8):
    # assume the first dimension is the number of experts
    var = torch.exp(logvar) + eps
    T = 1 / var  # precision of i-th Gaussian expert at point x
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1 / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var)

    return pd_mu, pd_logvar


def multivariate_product_of_experts(mu, logcov, eps=1e-8):
    # assume the first dimension is the number of experts
    # we also assume logcov is already in square form
    cov = torch.exp(logcov) + eps
    T = torch.inverse(cov)
    sum_T = torch.sum(T, dim=0)
    sum_T_inv = torch.inverse(T_cov)

    mT = torch.sum(torch.einsum('pbi,pbii->pbi', mu, T), dim=0)
    pd_mu = torch.einsum('bii,bii->bii', mT, sum_T_inv)
    
    pd_cov = sum_T_inv
    pd_logcov = torch.log(pd_cov + eps)

    return pd_mu, pd_logcov
