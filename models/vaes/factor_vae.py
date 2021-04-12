# Implementation of Factor Variational Auto-Encoder (paper at https://arxiv.org/pdf/1802.05983.pdf) based on pytorch

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import utils.nns as nns
from base import BaseVAE
from vae import VAE, ConvVAE


class FactorVAE(BaseVAE):
    """Class that implements the Factor Variational Auto-Encoder"""

    def __init__(self, n_input, n_hidden, dim_z, n_output, gamma, binary=True, **kwargs):
        """initialize neural networks
        :param gamma: weight for total correlation term in loss function
        """
        super(FactorVAE, self).__init__()
        self.dim_z = dim_z
        self.binary = binary
        self.gamma = gamma
        self.input_size = (n_input,)

        # VAE networks
        self.vae = VAE(n_input, n_hidden, dim_z, n_output, binary, **kwargs)

        # discriminator layers
        D_hidden_num = 3
        D_hidden_dim = 1000
        D_hidden_dims = [D_hidden_dim] * D_hidden_num
        D_act = nn.LeakyReLU
        D_act_args = {"negative_slope": 0.2, "inplace": False}
        D_output_dim = 2
        self.discriminator = nns.create_mlp(self.dim_z, D_hidden_dims,
                                            act_layer=D_act, act_args=D_act_args, norm=True)
        self.discriminator = nn.Sequential(
            self.discriminator,
            nn.Linear(D_hidden_dim, D_output_dim))

    def encode(self, x):
        """vae encode"""
        return self.vae.encode(x)

    def reparameterize(self, mu, logvar):
        """reparameterization trick"""
        return self.vae.reparameterize(mu, logvar)

    def decode(self, code):
        """vae deocde"""
        return self.vae.decode(code)

    def sample_latent(self, num, device, **kwargs):
        """vae sample latent"""
        return self.vae.sample_latent(num, device, **kwargs)

    def sample(self, num, device, **kwargs):
        """vae sample"""
        return self.vae.sample(num, device, **kwargs)

    def forward(self, input, no_dec=False):
        """autoencoder forward computation"""
        encoded = self.encode(input)
        mu, logvar = encoded
        z = self.reparameterize(mu, logvar) # latent variable z

        if no_dec:
            # no decoding
            return z

        return self.decode(z), encoded, self.discriminator(z)

    def decoded_to_output(self, decoded, **kwargs):
        """vae transform decoded result to output"""
        return self.vae.decoded_to_output(decoded, **kwargs)

    def reconstruct(self, input, **kwargs):
        """vae reconstruct"""
        return self.vae.reconstruct(input, **kwargs)

    def permute_dims(self, z):
        """permute separately each dimension of the z randomly in a batch
        :param z: [B x D] tensor
        :return: [B x D] tensor with each dim of D dims permuted randomly
        """
        B, D = z.size()
        # generate randomly permuted batch on each dimension
        permuted = []
        for i in range(D):
            ind = torch.randperm(B)
            permuted.append(z[:, i][ind].view(-1, 1))

        return torch.cat(permuted, dim=1)

    def disc_permute_z(self, z):
        """discriminator permute and forward computation"""
        z_perm = self.permute_dims(z).detach() # avoid updating
        Dz_perm = self.discriminator(z_perm)

        return Dz_perm

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (2))"""
        optim_part = kwargs['optim_part'] # the part to optimize

        if optim_part == 'vae':
            # update VAE
            decoded = inputs[0]
            encoded = inputs[1]
            Dz = inputs[2]
            x = inputs[3]
            mu, logvar = encoded

            # KL divergence term
            KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            if self.binary:
                # likelihood term under Bernolli MLP decoder
                MLD = F.binary_cross_entropy(decoded, x, reduction='sum').div(x.size(0))
            else:
                # likelihood term under Gaussian MLP decoder
                mu_o, logvar_o = decoded
                recon_x_distribution = Normal(
                    loc=mu_o, scale=torch.exp(0.5*logvar_o))
                MLD = -recon_x_distribution.log_prob(x).sum(1).mean()

            tc_loss = (Dz[:, :1] - Dz[:, 1:]).mean()

            return {
                "loss": KLD + MLD + self.gamma * tc_loss,
                "KLD": KLD,
                "MLD": MLD,
                "tc_loss": tc_loss}
        elif optim_part == 'discriminator':
            # update discriminator
            Dz = inputs[0]
            Dz_pperm = inputs[1]
            device = Dz.device

            ones = torch.ones(
                Dz.size(0), dtype=torch.long, requires_grad=False).to(device)
            zeros = torch.zeros(
                Dz.size(0), dtype=torch.long, requires_grad=False).to(device)

            D_tc_loss = 0.5 * (F.cross_entropy(Dz, zeros) +
                            F.cross_entropy(Dz_pperm, ones))

            return {"loss": D_tc_loss, "D_tc_loss": D_tc_loss}

        else:
            raise Exception("no such network to optimize: {}".format(optim_part))


class ConvFactorVAE(BaseVAE):
    """Class that implements the Factor Variational Auto-Encoder (based on CNN)"""

    def __init__(self, disc_hiddens=[1000, 1000, 1000],
                 gamma=30, input_size=(3, 64, 64),
                 kernel_sizes=[32, 32, 64, 64],
                 hidden_size=256, dim_z=32,
                 binary=True, **kwargs):
        """initialize neural networks
        :param disc_hiddens: list of int, numbers of hidden units of each layer in discriminator
        :param gamma: weight for total correlation term in loss function
        """
        super(ConvFactorVAE, self).__init__()
        self.gamma = gamma
        self.dim_z = dim_z
        self.binary = binary
        self.input_size = input_size
        self.hidden_size = hidden_size

        # VAE networks
        self.vae = ConvVAE(input_size, kernel_sizes, hidden_size, dim_z, binary, **kwargs)
        # inherit some attributes
        self.channel_sizes = self.vae.channel_sizes

        # discriminator networks
        D_act = nn.LeakyReLU
        D_act_args = {"negative_slope": 0.2, "inplace": False}
        D_output_dim = 2
        self.discriminator = nns.create_mlp(
            self.dim_z, disc_hiddens, act_layer=D_act, act_args=D_act_args)
        self.discriminator = nn.Sequential(
            self.discriminator,
            nn.Linear(disc_hiddens[-1], D_output_dim))

    def encode(self, x):
        """vae encode"""
        return self.vae.encode(x)

    def decode(self, code):
        """vae decode"""
        return self.vae.decode(code)

    def reparameterize(self, mu, logvar):
        """reparameterization trick"""
        return self.vae.reparameterize(mu, logvar)

    def forward(self, input, no_dec=False):
        """autoencoder forward computation"""
        encoded = self.encode(input)
        mu, logvar = encoded
        z = self.reparameterize(mu, logvar) # latent variable z

        if no_dec:
            # no decoding
            return z

        return self.decode(z), encoded, self.discriminator(z)

    def sample_latent(self, num, device, **kwargs):
        """vae sample latent"""
        return self.vae.sample_latent(num, device, **kwargs)

    def sample(self, num, device, **kwargs):
        """vae sample"""
        return self.vae.sample(num, device, **kwargs)

    def decoded_to_output(self, decoded, **kwargs):
        """vae transform decoded result to output"""
        return self.vae.decoded_to_output(decoded, **kwargs)

    def reconstruct(self, input, **kwargs):
        """vae reconstruct"""
        return self.vae.reconstruct(input, **kwargs)

    def permute_dims(self, z):
        """permute separately each dimension of the z randomly in a batch
        :param z: [B x D] tensor
        :return: [B x D] tensor with each dim of D dims permuted randomly
        """
        B, D = z.size()
        # generate randomly permuted batch on each dimension
        permuted = []
        for i in range(D):
            ind = torch.randperm(B)
            permuted.append(z[:, i][ind].view(-1, 1))

        return torch.cat(permuted, dim=1)

    def disc_permute_z(self, z):
        """discriminator permute and forward computation"""
        z_perm = self.permute_dims(z).detach()
        Dz_perm = self.discriminator(z_perm)

        return Dz_perm

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (2))"""
        optim_part = kwargs['optim_part'] # the part to optimize

        if optim_part == 'vae':
            # update VAE
            decoded = inputs[0]
            encoded = inputs[1]
            Dz = inputs[2]
            x = inputs[3]

            flat_input_size = np.prod(self.input_size)
            mu, logvar = encoded
            # KL divergence term
            KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            if self.binary:
                # likelihood term under Bernolli MLP decoder
                MLD = F.binary_cross_entropy(decoded.view(-1, flat_input_size),
                                             x.view(-1, flat_input_size),
                                             reduction='sum').div(x.size(0))
            else:
                # likelihood term under Gaussian MLP decoder
                mean_dec, logvar_dec = decoded
                recon_x_distribution = Normal(loc=mean_dec.view(-1, flat_input_size),
                                              scale=torch.exp(0.5*logvar_dec.view(-1, flat_input_size)))
                MLD = -recon_x_distribution.log_prob(x.view(-1, flat_input_size)).sum(1).mean()

            tc_loss = (Dz[:, :1] - Dz[:, 1:]).mean()

            return {
                "loss": KLD + MLD + self.gamma * tc_loss,
                "KLD": KLD,
                "MLD": MLD,
                "tc_loss": tc_loss}
        elif optim_part == 'discriminator':
            # update discriminator
            Dz = inputs[0]
            Dz_pperm = inputs[1]
            device = Dz.device

            ones = torch.ones(Dz.size(0), dtype=torch.long).to(device)
            zeros = torch.zeros(Dz.size(0), dtype=torch.long).to(device)

            D_tc_loss = 0.5 * (F.cross_entropy(Dz, zeros) +
                            F.cross_entropy(Dz_pperm, ones))

            return {"loss": D_tc_loss, "D_tc_loss": D_tc_loss}

        else:
            raise Exception("no such network to optimize: {}".format(optim_part))
