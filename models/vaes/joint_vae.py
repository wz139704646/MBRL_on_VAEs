# Implementation of Joint Variational Auto-Encoder (paper at https://arxiv.org/abs/1804.00104) based on pytorch
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

import utils.nns as nns
from base import BaseVAE


class ConvJointVAE(BaseVAE):
    """Class that implements Joint VAE (based on CNN version)"""
    EPS = 1e-12

    def __init__(self, temp=.67, latent_disc=[],
                 disc_capacity=None,
                 cont_capacity=(0, 5, 2e-4, 30),
                 input_size=(3, 64, 64),
                 kernel_sizes=[32, 32, 64, 64],
                 hidden_size=256, dim_z=32,
                 binary=True, **kwargs):
        """initialize neural networks
        :param temp: temperature for gumbel softmax distribution
        :param latent_disc: list, specify discrete latent varaibles.
            e.g. [10, 4, 3] (3 gumbel softmax variables of dimension 10, 4, 3).
            Default None, which represents no discrete variables
        :param disc_capacity: (min, max, delta, gamma) for capacity of discrete channels
            (not None if discrete variables exist)
        :param cont_capacity: (min, max, delta, gamma) for capacity of continuous channels
            (not None if continuous variables exist)
        :param dim_z: total latent variable dimension size
        """
        super(ConvJointVAE, self).__init__()
        self.temp = temp
        self.latent_disc = latent_disc
        self.input_size = input_size
        self.channel_sizes = [input_size[0]] + kernel_sizes
        self.hidden_size = hidden_size
        self.dim_z = dim_z
        self.binary = binary
        self.disc_capacity = disc_capacity
        self.cont_capacity = cont_capacity

        # calculate dimensions
        self.dim_disc = 0
        self.num_disc_latents = 0
        if self.latent_disc is not None:
            self.dim_disc = sum([dim for dim in latent_disc])
            self.num_disc_latents = len(latent_disc)
            self.disc_cap_current = self.disc_capacity[0]
        self.dim_cont = self.dim_z - self.dim_disc
        if self.dim_cont > 0:
            self.cont_cap_current = self.cont_capacity[0]

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
        # hidden units to latent variables
        if self.dim_cont > 0:
            # continuous variables, Gaussian MLP
            self.fc_mean = nn.Linear(hidden_size, self.dim_cont)
            self.fc_logvar = nn.Linear(hidden_size, self.dim_cont)
        if self.dim_disc > 0:
            # discrete variables, Gumbel MLP
            fc_alphas = []
            for disc_dim in self.latent_disc:
                fc_alphas.append(nn.Linear(self.hidden_size, disc_dim))
            self.fc_alphas = nn.ModuleList(fc_alphas)

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
        """CNN + Gaussian/Gumbel MLP encoder
        :return: tuple (len=2) of list, which represents encoded continuous variables
            and encoded discrete variables (one of them may be empty)
        """
        features = self.conv_encoder(x)
        features = features.view(-1, self.flat_conv_output_size)
        h = self.features_to_hidden(features)

        encoded_cont = []
        encoded_disc = []
        if self.dim_cont > 0:
            encoded_cont = [self.fc_mean(h), self.fc_logvar(h)]
        if self.dim_disc > 0:
            for fc_alpha in self.fc_alphas:
                encoded_disc.append(F.softmax(fc_alpha(h), dim=1))

        return encoded_cont, encoded_disc

    def decode(self, code):
        """CNN decoder"""
        features = self.latent_to_features(code)
        features = features.view(-1, *self.conv_output_size)

        if self.binary:
            return self.conv_decoder(features)

        return self.conv_decoder(features), self.conv_decoder_logvar(features)

    def reparameterize(self, encoded_cont, encoded_disc):
        """reparameterization trick.
            For continuous variables, use normal reparameterization.
            For discrete variables, use Gumbel softmax reparameterization.
        """
        code = []

        if self.dim_cont > 0:
            mu, logvar = encoded_cont
            code.append(self.sample_normal(mu, logvar))
        if self.dim_disc > 0:
            for alpha in encoded_disc:
                disc_sample = self.sample_gumbel_softmax(alpha)
                code.append(disc_sample)

        # concatenate continuous and discrete variables into one code
        return torch.cat(code, dim=1)

    def sample_normal(self, mu, logvar):
        """sample from normal distribution using reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            return mu + std * eps
        else:
            # reconstruction mode, use mean value
            return mu

    def sample_gumbel_softmax(self, alpha):
        """sample from a gumbel-softmax distribution using reparameterization trick"""
        if self.training:
            unif = torch.rand(alpha.size())
            gumbel = -torch.log(-torch.log(unif + ConvJointVAE.EPS) + ConvJointVAE.EPS)
            gumbel = gumbel.to(alpha.device)
            # reparameterize to create softmax sample
            log_alpha = torch.log(alpha + ConvJointVAE.EPS)
            logit = (log_alpha + gumbel) / self.temp

            return F.softmax(logit, dim=1)
        else:
            # reconstruction mode, use most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # set the max alpha index 1
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)

            return one_hot_samples.to(alpha.device)

    def forward(self, input):
        """autoencoder forward computation"""
        encoded = self.encode(input)
        z = self.reparameterize(*encoded)

        return self.decode(z), encoded

    def sample_latent(self, num, device, **kwargs):
        """sample from latent space"""
        z = []

        if self.dim_cont > 0:
            cont_z = torch.randn(num, self.dim_cont)

            z.append(cont_z)
        if self.dim_disc > 0:
            for disc_dim in self.latent_disc:
                # for each discrete variable, sample from uniform prior
                disc_z = torch.zeros(num, disc_dim)
                disc_z[np.arange(num), np.random.randint(0, disc_dim, num)] = 1.

                z.append(disc_z)

        z = torch.cat(z, dim=1).to(device)

        return z

    def sample(self, num, device, **kwargs):
        """sample from latent space and return the decoded output"""
        z = self.sample_latent(num, device, **kwargs)
        samples = self.decode(z)

        if not self.binary:
            # Gaussian
            mean_dec, logvar_dec = samples
            samples = self.sample_normal(
                torch.flatten(mean_dec, start_dim=1),
                torch.flatten(logvar_dec, start_dim=1)
            ).view(-1, *self.input_size)

        return samples

    def decoded_to_output(self, decoded, **kwargs):
        """return the output for given decoded result"""
        if self.binary:
            # Bernoulli, directly return
            return decoded.clone().detach()

        # Gaussian
        mean_dec, logvar_dec = decoded
        return self.sample_normal(
            torch.flatten(mean_dec, start_dim=1),
            torch.flatten(logvar_dec, start_dim=1)
        ).view(-1, *self.input_size)

    def reconstruct(self, input, **kwargs):
        """reconstruct from the input"""
        decoded = self.forward(input)[0]

        if not self.binary:
            # Gaussian
            mean_dec, logvar_dec = decoded
            decoded = self.sample_normal(
                torch.flatten(mean_dec, start_dim=1),
                torch.flatten(logvar_dec, start_dim=1)
            ).view(-1, *self.input_size)

        # Bernoulli, directly return
        return decoded

    def loss_function(self, *inputs, **kwargs):
        """loss functin described in the paper (eq. (7))"""
        decoded = inputs[0]
        encoded = inputs[1]
        x = inputs[2]

        flat_input_size = np.prod(self.input_size)
        encoded_cont, encoded_disc = encoded

        # KL divergence term
        KL_cont_loss = torch.tensor(0.)
        KL_disc_loss = torch.tensor(0.)
        cont_capacity_loss = torch.tensor(0.)
        disc_capacity_loss = torch.tensor(0.)

        if self.dim_cont > 0:
            # KL divergence term of continuous variables
            mu, logvar = encoded_cont
            KL_cont_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
            _, _, _, cont_gamma = self.cont_capacity

            cont_capacity_loss = cont_gamma * torch.abs(KL_cont_loss - self.cont_cap_current)
        if self.dim_disc > 0:
            # KL divergence term of discrete variables
            KL_disc_losses = [self._KL_discrete_loss(alpha) for alpha in encoded_disc]
            KL_disc_loss = torch.sum(torch.cat(KL_disc_losses))
            _, _, _, disc_gamma = self.disc_capacity

            disc_capacity_loss = disc_gamma * torch.abs(KL_disc_loss - self.disc_cap_current)

        KL_loss = KL_cont_loss + KL_disc_loss

        # reconstruction loss
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

        return {
            "loss": MLD + cont_capacity_loss + disc_capacity_loss,
            "MLD": MLD,
            "KL_loss": KL_loss,
            "KL_cont_loss": KL_cont_loss,
            "KL_disc_loss": KL_disc_loss,
            "cont_capacity_loss": cont_capacity_loss,
            "disc_capacity_loss": disc_capacity_loss
        }

    def _KL_discrete_loss(self, alpha):
        """calculate the KL divergence between a categorical distribution
            and a uniform categorical distribution
        """
        dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(dim)]).to(alpha.device)

        # calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + ConvJointVAE.EPS), dim=1)
        # mean over batch
        neg_entropy = torch.mean(neg_entropy, dim=0)

        return log_dim + neg_entropy

    def update_step(self):
        """update capacity after each step"""
        if self.dim_cont > 0:
            # linearly increase capacity of continuous channels
            _, cont_max, cont_delta, _ = self.cont_capacity
            # increase continuous capacity without exceeding cont_max
            self.cont_cap_current = min(cont_max, self.cont_cap_current + cont_delta)
        if self.dim_disc > 0:
            # linearly increase capacity of discrete channels
            _, disc_max, disc_delta, _ = self.disc_capacity
            # increase discrete capacity without exceeding disc_max or theoretical
            # maximum (i.e. sum of log of dimension of each discrete variable)
            self.disc_cap_current = min(disc_max, self.disc_cap_current + disc_delta)
            disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.latent_disc])
            self.disc_cap_current = min(self.disc_cap_current, disc_theoretical_max)
