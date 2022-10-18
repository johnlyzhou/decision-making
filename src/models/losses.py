import torch


def gaussian_nll(x, x_hat, scale=1):
    predicted_dist = torch.distributions.Normal(x_hat, scale)
    batch_size = x.shape[0]
    return -predicted_dist.log_prob(x).view(batch_size, -1).sum(dim=1)


def kl_divergence(mu, log_var):
    return torch.mean(0.5 * torch.sum(log_var.exp() - log_var + mu ** 2 - 1, dim=1), dim=0)
