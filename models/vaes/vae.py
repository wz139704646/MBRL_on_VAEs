# Implementation of Variational Auto-Encoder (paper at https://arxiv.org/abs/1312.6114) based on pytorch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import utils.nns as nns
from base import BaseVAE


class VAE(BaseVAE):
    """Class that implements Variational Auto-Encoder (classic version based on MLP)"""

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
        
        return MLD + KLD


class ConvVAE(BaseVAE):
    """Class that implements Variational Auto-Encoder (based on CNN version)"""

    def __init__(self, input_size=(3, 64, 64),
                 kernel_sizes=[32, 32, 64, 64],
                 hidden_size=256, dim_z=32,
                 binary=True, **kwargs):
        """initialize neural networks
        :param input_size: C x H x W
        :param kernel_sizes: number of channels in the kernels
        :param hidden_size: size of hidden layer
        :param dim_z: dimension of latent variable z
        :param binary: whether the data is binary
        """
        super(ConvVAE, self).__init__()
        self.input_size = input_size
        self.channel_sizes = [input_size[0]] + kernel_sizes
        self.hidden_size = hidden_size
        self.dim_z = dim_z
        self.binary = binary

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
        """CNN + Gaussian MLP encoder"""
        features = self.conv_encoder(x)
        features = features.view(-1, self.flat_conv_output_size)
        h = self.features_to_hidden(features)

        return self.fc_mean(h), self.fc_logvar(h)
    
    def decode(self, code):
        """CNN decoder"""
        features = self.latent_to_features(code)
        features = features.view(-1, *self.conv_output_size)
        if self.binary:
            return self.conv_decoder(features)
        
        return self.conv_decoder(features), self.conv_decoder_logvar(features) 

    def reparameterize(self, mu, logvar):
        """reparameterization trick"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # noise epsilon

        return mu + eps * std

    def forward(self, input):
        """autoencoder forward computation"""
        encoded = self.encode(input)
        mu, logvar = encoded
        z = self.reparameterize(mu, logvar)

        return self.decode(z), encoded

    def sample(self, num, device, **kwargs):
        """sample from latent space and return the decoded output"""
        z = torch.randn(num, self.dim_z).to(device)
        samples = self.decode(z)

        if not self.binary:
            # Gaussian
            mean_dec, logvar_dec = samples
            samples = self.reparameterize(
                torch.flatten(mean_dec, start_dim=1),
                torch.flatten(logvar_dec, start_dim=1)
            ).view(-1, *self.input_size)
        
        # Bernoulli, directly return
        return samples

    def reconstruct(self, input, **kwargs):
        """reconstruct from the input"""
        decoded = self.forward(input)[0]

        if not self.binary:
            # Gaussian
            mean_dec, logvar_dec = decoded
            decoded = self.reparameterize(
                torch.flatten(mean_dec, start_dim=1),
                torch.flatten(logvar_dec, start_dim=1)
            ).view(-1, *self.input_size)
        
        # Bernoulli, directly return
        return decoded

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (10))"""
        decoded = inputs[0]
        x = inputs[1]
        encoded = inputs[2]

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
        
        return KLD + MLD