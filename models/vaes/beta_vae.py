# Implementation of beta-VAE (paper at https://openreview.net/forum?id=Sy2fzU9gl) based on pytorch

import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions.normal import Normal

from vae import VAE, ConvVAE


class BetaVAE(VAE):
    """Class that implements beta Variational Auto-Encoder"""

    def __init__(self, n_input, n_hidden, dim_z, n_output, beta, binary=True, **kwargs):
        """initialize neural networks
        :param beta: coefficient applied in loss function for KL term
        """
        super(BetaVAE, self).__init__(n_input, n_hidden, dim_z, n_output, binary, **kwargs)
        self.beta = beta # additional coef compared to original VAE

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (10))"""
        decoded = inputs[0]
        encoded = inputs[1]
        x = inputs[2]

        mu, logvar = encoded
        # KL divergence term
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        if self.binary:
            # likelihood term under Bernolli MLP decoder
            MLD = F.binary_cross_entropy(decoded, x, reduction='sum').div(x.size(0))
        else:
            # likelihood term under Gaussian MLP decoder
            mu_o, logvar_o = decoded
            recon_x_distribution = Normal(loc=mu_o, scale=torch.exp(0.5*logvar_o))
            MLD = -recon_x_distribution.log_prob(x).sum(1).mean()

        return {"loss": MLD + self.beta * KLD, "KLD": KLD, "MLD": MLD}


class ConvBetaVAE(ConvVAE):
    """Class that implements Variational Auto-Encoder (based on CNN version)"""

    def __init__(self, input_size=(3, 64, 64),
                 kernel_sizes=[32, 32, 64, 64],
                 hidden_size=256, dim_z=32,
                 beta=3., binary=True, **kwargs):
        """initialize neural networks
        :param beta: coefficient applied in loss fucntion for KL term
        """
        super(ConvBetaVAE, self).__init__(input_size, kernel_sizes, hidden_size, dim_z, binary, **kwargs)
        self.beta = beta

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (10))"""
        decoded = inputs[0]
        encoded = inputs[1]
        x = inputs[2]

        flat_input_size = np.prod(self.input_size)
        mu, logvar = encoded
        # KL divergence term
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        if self.binary:
            # likelihood term under Bernoulli distribution
            MLD = F.binary_cross_entropy(decoded.view(-1, flat_input_size),
                                         x.view(-1, flat_input_size),
                                         reduction='sum').div(x.size(0))
        else:
            # likelihood term under Gaussian distribution
            mean_dec, logvar_dec = decoded
            recon_x_distribution = Normal(loc=mean_dec.view(-1, flat_input_size),
                                          scale=torch.exp(0.5*logvar_dec.view(-1, flat_input_size)))
            MLD = -recon_x_distribution.log_prob(x.view(-1, flat_input_size)).sum(1).mean()

        return {"loss": self.beta * KLD + MLD, "KLD": KLD, "MLD": MLD}