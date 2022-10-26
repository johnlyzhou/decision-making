import torch
from torch import tensor, nn


def gaussian_nll(x: tensor, x_hat: tensor, scale: int = 1):
    predicted_dist = torch.distributions.Normal(x_hat, scale)
    batch_size = x.shape[0]
    return -predicted_dist.log_prob(x).view(batch_size, -1).sum(dim=1)


def kl_divergence(mu: tensor, log_var: tensor):
    return torch.mean(0.5 * torch.sum(log_var.exp() - log_var + mu ** 2 - 1, dim=1), dim=0)


def sigmoid_loss(x: tensor, params: tensor):
    loss = nn.MSELoss()
    x_range = torch.arange(0, x.size(dim=2))
    x_hat = params[:, :, 0] + (1 - 2 * params[:, :, 0]) / (1 + torch.exp(-params[:, :, 1] * (x_range - params[:, :, 2])))
    return loss(x, torch.unsqueeze(x_hat, 1))
