import numpy as np
from gym.envs.mujoco.swimmer import SwimmerEnv


from envs.virtual_env import BaseModelBasedEnv


class OriginalSwimmerEnv(SwimmerEnv, BaseModelBasedEnv):

    def mb_step(self, states, actions, next_states):
        return None, np.zeros([states.shape[0], 1], dtype=np.bool)
