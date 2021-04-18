# the network to learn termination
import torch
import torch.nn as nn
from typing import List

from models.mlp import MLP


class TerminationFn(nn.Module):
    """class that learn the environment termination from (state, next_state, action) pairs"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], **kwargs):
        """initialize networks"""
        super(TerminationFn, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        mlp_kwargs = kwargs.copy()
        mlp_kwargs['activation'] = kwargs.get('activation', 'relu')
        mlp_kwargs['last_activation'] = 'sigmoid'
        self.input_dim = state_dim * 2 + action_dim # (state, action, next_state) pair
        self.output_dim = 1
        self.net = MLP(self.input_dim, self.output_dim, hidden_dims, **mlp_kwargs)
        self.criterion = nn.BCELoss()
        self.boundary = 0.5

    def forward(self, states, actions, next_states):
        """forward computation"""
        # concate (s, a, s') to one vector
        x = torch.cat([states, actions, next_states], dim=1)

        return self.net(x)

    def done(self, model_output, to_bool=True):
        """convert model_output to bool/float dones info"""
        if to_bool:
            return model_output.ge(self.boundary)
        else:
            return model_output.ge(self.boundary).float()

    def loss_function(self, *inputs, **kwargs):
        """bce loss function"""
        pred_terms = inputs[0]
        real_dones = inputs[1]

        return {'loss': self.criterion(pred_terms, real_dones.float())}

