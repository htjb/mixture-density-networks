import numpy as np
import torch
import torch.nn as nn

"""
Inspired by https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py
and https://mikedusenberry.com/mixture-density-networks.
"""

class MDN(nn.Module):

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians

        # this is the prior probability of each gaussian
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )

        # these are the parameters of the gaussians
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, x):
        """
        Forward pass through the MDN layers
        """
        pi = self.pi(x)
        # hmm not sure what these two lines are doing...
        # ohh this is passing x through the linear sigma layer which 
        # is actually predicting ln(sigma) and then reshaping it
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)

        # same as above but for mu
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.out_features)

        return pi, sigma, mu
    
def gaussian_probability(sigma, mu, target):
    """
    Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = (1.0 / torch.sqrt(2.0 * np.pi * sigma)) * \
        torch.exp(-0.5 * (target - mu)**2 / sigma**2)
    # torch.prod takes the product and reduces across dimension 2
    return torch.prod(ret, 2)

def mdn_loss(pi, sigma, mu, target):
    """
    Calculates the error, given the MoG parameters and the target. Loss is
    negative log likelihood of the data given the MoG parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)

def sample(pi, sigma, mu):
    """Sample from the MoG"""

    # pick which mode to sample from based on the learned weights pi
    pis = torch.distributions.Categorical(pi).sample().view(pi.size(0), 1, 1)

    gaussian_noise = torch.randn((sigma.size(2), sigma.size(0)), 
                                 requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.gather(1, pis).detach().squeeze()
    return (mean_samples + variance_samples * gaussian_noise).transpose(0, 1)

