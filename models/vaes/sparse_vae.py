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
                 alpha, delta_c, binary=True, **kwargs):
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
        self.delta_c = delta_c # c warm-up

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
        return torch.randn(num, self.dim_z).to(device)

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

    def update_epoch(self):
        """warm-up strategy gradually increasing c during training"""
        self.c += self.delta_c


class ConvSparseVAE(BaseVAE):
    def __init__(self, alpha, c=50, delta_c=1.0e-3,
                 beta=0.1, delta_beta=0,
                 input_size=(3, 64, 64),
                 kernel_sizes=[32, 32, 64, 64],
                 hidden_size=256, dim_z=32,
                 binary=True, **kwargs):
        """initialize neural networks
        :param alpha: prior sparsity
        :param input_size: C x H x W
        :param kernel_sizes: number of channels in the kernels
        :param hidden_size: size of hidden layer
        :param dim_z: dimension of latent variable z
        :param binary: whether the data is binary
        """
        super(ConvSparseVAE, self).__init__()
        self.input_size = input_size
        self.channel_sizes = [input_size[0]] + kernel_sizes
        self.hidden_size = hidden_size
        self.dim_z = dim_z
        self.binary = binary
        self.alpha = alpha
        self.c = c
        self.delta_c = delta_c
        self.beta = beta # weight for prior
        self.delta_beta = delta_beta

        # initialize encoder layers
        self.conv_encoder = nns.create_cnn2d(
            input_size[0], kernel_sizes, (4,4), 2, 1)
        # encoder conv layer output size
        conv_out_size = int(input_size[-1] / (2 ** len(kernel_sizes)))
        self.conv_output_size = (self.channel_sizes[-1], conv_out_size, conv_out_size)
        self.flat_conv_output_size = np.prod(self.conv_output_size)

        # layers transfer features to hidden units
        self.features_to_hidden = nns.create_mlp(
            self.flat_conv_output_size, [hidden_size])

        # Gaussian MLP
        self.fc_mean = nn.Linear(hidden_size, dim_z)
        self.fc_logvar = nn.Linear(hidden_size, dim_z)
        self.fc_logspike = nn.Linear(hidden_size, dim_z)

        # layers transform latent variables to features (via hidden units)
        self.latent_to_features = nns.create_mlp(
            dim_z, [hidden_size, self.flat_conv_output_size])

        # initialize decoder layers
        self.conv_decoder = nns.create_transpose_cnn2d(
            self.channel_sizes[-1], self.channel_sizes[-2:0:-1], (4,4), 2, 1, 0)
        self.conv_decoder = nn.Sequential(
            self.conv_decoder,
            nn.ConvTranspose2d(self.channel_sizes[1], self.channel_sizes[0],
                               (4,4), stride=2, padding=1))
        # the final layer
        if binary:
            # for binary data use sigmoid activation function
            self.conv_decoder = nn.Sequential(
                self.conv_decoder,
                nn.Sigmoid())
        else:
            # for non-binary data use extra Gaussian MLP (add a logvar transposed cnn)
            self.conv_decoder_logvar = nns.create_transpose_cnn2d(
                self.channel_sizes[-1], self.channel_sizes[-2:0:-1], (4,4), 2, 1, 0)
            self.conv_decoder_logvar = nn.Sequential(
                self.conv_decoder,
                nn.ConvTranspose2d(self.channel_sizes[1], self.channel_sizes[0],
                                   (4,4), stride=2, padding=1))

    def encode(self, x):
        """recognition function"""
        features = self.conv_encoder(x)
        features = features.view(-1, self.flat_conv_output_size)
        h = self.features_to_hidden(features)

        return self.fc_mean(h), self.fc_logvar(h), -F.relu(-self.fc_logspike(h))

    def decode(self, code):
        """likelihood function"""
        features = self.latent_to_features(code)
        features = features.view(-1, *self.conv_output_size)

        if self.binary:
            return self.conv_decoder(features)

        return self.conv_decoder(features), self.conv_decoder_logvar(features)

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

    def forward(self, x):
        """autoencoder forward computation"""
        encoded = self.encode(x)
        mu, logvar, logspike = encoded
        z = self.reparameterize(mu, logvar, logspike)

        return self.decode(z), encoded

    def sample_latent(self, num, device, **kwargs):
        """sample from latent space and return the codes"""
        return torch.randn(num, self.dim_z).to(device)

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

        flat_input_size = np.prod(self.input_size)
        mu, logvar, logspike = encoded
        if self.binary:
            MLD = F.binary_cross_entropy(decoded.view(-1, flat_input_size),
                                         x.view(-1, flat_input_size),
                                         reduction="sum").div(x.size(0))
        else:
            mean_dec, logvar_dec = decoded
            recon_x_distribution = Normal(loc=mean_dec.view(-1, flat_input_size),
                                          scale=torch.exp(0.5*logvar_dec.view(-1, flat_input_size)))
            MLD = -recon_x_distribution.log_prob(x.view(-1, flat_input_size)).sum(1).mean()

        spike = torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6)
        prior1 = -0.5 * torch.sum(spike * (1 + logvar - mu.pow(2) - logvar.exp()))
        prior21 = (1 - spike) * (torch.log((1 - spike) / (1 - self.alpha)))
        prior22 = spike * torch.log(spike / self.alpha)
        prior2 = torch.sum(prior21 + prior22)
        PRIOR = prior1 + prior2
        PRIOR = PRIOR.div(x.size(0))

        return {
            "loss": MLD + self.beta * PRIOR,
            "MLD": MLD,
            "prior1": prior1,
            "prior2": prior2,
            "PRIOR": PRIOR}

    def update_epoch(self):
        """warm-up strategy gradually increasing c and beta during training"""
        self.c += self.delta_c
        self.beta += self.delta_beta
