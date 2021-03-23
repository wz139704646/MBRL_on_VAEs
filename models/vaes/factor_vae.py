# Implementation of Factor Variational Auto-Encoder (paper at https://arxiv.org/pdf/1802.05983.pdf) based on pytorch

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import utils.nns as nns
from vae import VAE, ConvVAE


class FactorVAE(VAE):
    """Class that implements the Factor Variational Auto-Encoder"""

    def __init__(self, n_input, n_hidden, dim_z, n_output, gamma, binary=True, **kwargs):
        """initialize neural networks
        :param gamma: weight for total correlation term in loss function
        """
        super(FactorVAE, self).__init__(n_input, n_hidden, dim_z, n_output, binary, **kwargs)
        self.gamma = gamma

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

    def forward(self, input, no_dec=False):
        """autoencoder forward computation"""
        encoded = self.encode(input)
        mu, logvar = encoded
        z = self.reparameterize(mu, logvar) # latent variable z

        if no_dec:
            # no decoding
            return z.squeeze()

        return self.decode(z), encoded, z

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

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (2))"""
        optim_index = kwargs['optim_index'] # index of optimizer

        if optim_index == 0:
            # update VAE
            decoded = inputs[0]
            encoded = inputs[1]
            z = inputs[2]
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

            Dz = self.discriminator(z)
            tc_loss = (Dz[:, :1] - Dz[:, 1:]).mean()

            return {
                "loss": KLD + MLD + self.gamma * tc_loss,
                "KLD": KLD,
                "MLD": MLD,
                "tc_loss": tc_loss}
        elif optim_index == 1:
            # update discriminator
            z = inputs[0].detach()
            z_prime = inputs[1]
            device = z.device

            ones = torch.ones(
                z.size(0), dtype=torch.long, requires_grad=False).to(device)
            zeros = torch.zeros(
                z.size(0), dtype=torch.long, requires_grad=False).to(device)

            Dz = self.discriminator(z)
            z_pperm = self.permute_dims(z_prime).detach() # avoid updating VAE again
            Dz_pperm = self.discriminator(z_pperm)
            D_tc_loss = 0.5 * (F.cross_entropy(Dz, zeros) +
                            F.cross_entropy(Dz_pperm, ones))

            return {"loss": D_tc_loss, "D_tc_loss": D_tc_loss}