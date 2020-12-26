# Implementation of Variational Auto-Encoder (paper at https://arxiv.org/abs/1312.6114) based on pytorch

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

from base import BaseVAE


class VAE(BaseVAE):
    """Class that implements Variational Auto-Encoder"""

    def __init__(self, n_input, n_hidden, dim_z, n_output, binary=True, **kwargs):
        """initialize neural networks
        :param binary: whether the input is binary data (determine which decoder to use)
        """
        super(VAE, self).__init__()
        self.dim_z = dim_z
        self.binary = binary

        # encoder layers, use Gaussian MLP as encoder
        self.fc1 = nn.Linear(n_input, n_hidden) # the first layer
        self.fc2_mean = nn.Linear(n_hidden, dim_z) # the second layer to compute mu
        self.fc2_var = nn.Linear(n_hidden, dim_z) # the second layer to compute sigma

        # decoder layers
        self.fc3 = nn.Linear(dim_z, n_hidden) # the third layer
        if binary:
            # binary input data, use Bernoulli MLP as decoder
            self.fc4 = nn.Linear(n_hidden, n_output) # the fourth layer
        else:
            # not binary data, use Gaussian MLP as decoder
            self.fc4_mean = nn.Linear(n_hidden, n_output)
            self.fc4_var = nn.Linear(n_hidden, n_output)

    def encode(self, x):
        """Gaussian MLP encoder"""
        h1 = F.tanh(self.fc1(x))

        return self.fc2_mean(h1), self.fc2_var(h1)

    def reparameterize(self, mu, logvar):
        """reparameterization trick"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # noise epsilon

        return mu + eps * std

    def decode(self, code):
        """Guassian/Bernoulli MLP decoder"""
        h3 = F.tanh(self.fc3(code))

        if self.binary:
            # binary input data, use Bernoulli MLP decoder
            return torch.sigmoid(self.fc4(h3))
        
        # otherwise use Gaussian MLP decoder
        return self.fc4_mean(h3), self.fc4_var(h3)

    def sample(self, num, device, **kwargs):
        """sample from latent space and return the decoded output"""
        z = torch.randn(num, self.dim_z).to(device)
        samples = self.decode(z)

        if not self.binary:
            # Gaussian
            mu_o, logvar_o = samples
            samples = self.reparameterize(mu_o, logvar_o)

        # otherwise Bernoulli
        return samples

    def forward(self, input):
        """autoencoder forward computation"""
        encoded = self.encode(input)
        mu, logvar = encoded
        z = self.reparameterize(mu, logvar) # latent variable z

        return self.decode(z), encoded

    def reconstruct(self, input, **kwargs):
        """reconstruct from the input"""
        decoded = self.forward(input)[0]
        
        if not self.binary:
            # use Guassian, smaple from decoded Gaussian distribution(use reparameterization trick)
            mu_o, logvar_o = decoded
            decoded = self.reparameterize(mu_o, logvar_o)

        # use Bernoulli, can directly return as the reconstructed result
        return decoded

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (10))"""
        decoded = inputs[0]
        x = inputs[1]
        encoded = inputs[2]

        mu, logvar = encoded
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence term
        if self.binary:
            # loss under Bernolli MLP decoder
            BCE = F.binary_cross_entropy(decoded, x, reduction='sum') # likelyhood term
            return BCE + KLD

        # otherwise, return loss under Gaussian MLP decoder
        mu_o, logvar_o = decoded
        recon_x_distribution = Normal(loc=mu_o, scale=torch.exp(0.5*logvar_o))
        MLD = -torch.sum(recon_x_distribution.log_prob(x))
        return MLD + KLD

