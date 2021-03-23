# Implementation of Variational Sparse Coding (also called sparse VAE) (paper at https://openreview.net/forum?id=SkeJ6iR9Km)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import utils.nns as nns
from base import BaseVAE


class SparseVAE(BaseVAE):
    """Class that implements the Sparse VAE (Variational Sparse Coding)"""

    def __init__(self, input_size, hidden_sizes, dim_z,
                 alpha, binary=True, **kwargs):
        """initialize neural networks
        :param input_size: dimension of the input
        :param hidden_sizes: number of hidden units for each hidden layer
        :param dim_z: dimension of the latent space
        :param alpha: prior sparsity
        :param binary: whether use binary data
        """
        super(SparseVAE, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dim_z = dim_z
        self.binary = binary
        self.alpha = alpha
        self.c = 50.0 # for reparametrization

        # encoder layers
        self.fc1n = nns.create_mlp(
            input_size, hidden_sizes, act_layer=nn.ReLU, act_args={}, norm=False)
        # self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # self.fc1n = utils.create_mlp(hidden_sizes, hidden_sizes[1:])
        self.fc2_mean = nn.Linear(hidden_sizes[-1], dim_z)
        self.fc2_logvar = nn.Linear(hidden_sizes[-1], dim_z)
        self.fc2_logspike = nn.Linear(hidden_sizes[-1], dim_z)

        # decoder layers
        self.fc3n = nns.create_mlp(
            dim_z, hidden_sizes[-1::-1], act_layer=nn.ReLU, act_args={}, norm=False)
        self.fc4 = nn.Linear(hidden_sizes[0], input_size)
        if not binary:
            # extra Gaussian MLP
            self.fc4_logvar = nn.Linear(hidden_sizes[0], input_size)


    def encode(self, x):
        """recognition function"""
        h = self.fc1n(x)

        return self.fc2_mean(h), self.fc2_logvar(h), -F.relu(-self.fc2_logspike(h))

    def reparameterize(self, mu, logvar, logspike):
        """reparameterization"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        gaussian = mu + eps * std
        if logspike is None:
            # Gaussian reparameterization
            return gaussian

        eta = torch.randn_like(std)
        selection = F.sigmoid(self.c*(eta + logspike.exp() - 1))

        return selection * gaussian

    def decode(self, code):
        """likelihood function"""
        h3 = self.fc3n(code)

        if self.binary:
            return torch.sigmoid(self.fc4(h3))

        return self.fc4(h3), self.fc4_logvar(h3)

    def forward(self, x):
        """autoencoder forward computation"""
        encoded = self.encode(x)
        mu, logvar, logspike = encoded
        z = self.reparameterize(mu, logvar, logspike)

        return self.decode(z), encoded

    def sample_latent(self, num, device, **kwargs):
        """sample from latent space and return the codes"""
        return torch.randn(num, self.dim_z)

    def sample(self, num, device, **kwargs):
        """sample from latent space and return the decoded output"""
        z = self.sample_latent(num, device, **kwargs)
        samples = self.decode(z)

        if not self.binary:
            mu, logvar = samples
            samples = self.reparameterize(mu, logvar, None)

        return samples

    def decoded_to_output(self, decoded, **kwargs):
        """return the output for given decoded result"""
        if self.binary:
            return decoded.clone().detach()

        # Gaussian
        mean, logvar = decoded
        return self.reparameterize(mean, logvar, None)

    def reconstruct(self, x, **kwargs):
        """reconstruct from the input"""
        decoded = self.forward(x)[0]

        return self.decoded_to_output(decoded, **kwargs)

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (11))"""
        decoded = inputs[0]
        encoded = inputs[1]
        x = inputs[2]

        mu, logvar, logspike = encoded
        if self.binary:
            MLD = F.binary_cross_entropy(decoded, x, reduction="sum").div(x.size(0))
        else:
            mu_dec, logvar_dec = decoded
            recon_x_distribution = Normal(loc=mu_dec, scale=torch.exp(0.5*logvar_dec))
            MLD = -recon_x_distribution.log_prob(x).sum(1).mean()

        spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6)
        PRIOR = -0.5 * torch.sum(spike * (1 + logvar - mu.pow(2) - logvar.exp())) + \
                       torch.sum((1-spike) * (torch.log((1-spike) / (1-self.alpha))) + \
                                 spike * torch.log(spike/self.alpha))
        PRIOR = PRIOR.div(x.size(0))

        return {"loss": MLD + PRIOR, "MLD": MLD, "PRIOR": PRIOR}

    def update_epoch(self, delta):
        """warm-up strategy gradually increasing c during training"""
        self.c += delta
