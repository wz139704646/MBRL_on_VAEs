from torch import nn
from abc import abstractmethod


class BaseVAE(nn.Module):
    """Base class for different VAE that describes basic structure"""

    def __init__(self):
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, code):
        raise NotImplementedError

    def sample_latent(self, num, device, **kwargs):
        raise NotImplementedError

    def sample(self, num, device, **kwargs):
        raise NotImplementedError

    def decoded_to_output(self, decoded, **kwargs):
        raise NotImplementedError

    def reconstruct(self, input, **kwargs):
        raise NotImplementedError

    def update_step(self):
        """updates after each train step"""
        pass

    def update_epoch(self):
        """updates after each train epoch"""
        pass

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass
