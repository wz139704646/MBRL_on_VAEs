import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch

from .actor import Actor
from .critic import QCritic
from .dynamics import RDynamics, EnsembleRDynamics
from .term_fn import TerminationFn


setattr(torch, 'identity', lambda x: x)
setattr(torch, 'swish', lambda x: x * torch.sigmoid(x))