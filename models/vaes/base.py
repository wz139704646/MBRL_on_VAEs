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

    def sample(self, num, device, **kwargs):
        raise NotImplementedError

    def reconstruct(self, input, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass
