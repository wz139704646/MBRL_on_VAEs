import math
import torch
import logging

from envs import *
from utils.wrappers_test import make_atari, wrap_deepmind, wrap_pytorch


def collect_env_data_gym(cfg):
    """collect data from observations of atari games"""
    env = wrap_pytorch(wrap_deepmind(make_atari(cfg.env_name), scale=True))
    obs = env.reset()  # initial observation
    data = []  # dataset buffer
    done = False
    for i in range(cfg.dataset_size):
        """for simplification, gym data collecting all use random policy"""
        # add the last frame in the queue to buffer
        data.append(torch.from_numpy(
            obs).squeeze()[-3:].unsqueeze(dim=0))
        ac = env.action_space.sample()  # use random policy
        if not done:
            obs, _, done, _ = env.step(ac)
        else:
            obs = env.reset()
            done = False

        if (i+1) % 100 == 0:
            logging.info("sampled {} frames".format(i+1))

    env.close()
    return torch.cat(data, dim=0)


def collect_env_data_custom(cfg):
    """collect data from observations of cutom env"""
    # initialize environment
    env_init_args = cfg.custom_args["env_init_args"]
    env = eval(cfg.env_name)(**env_init_args)

    """for simplification, custom env data collecting all use traversal policy"""
    # traverse and collect data
    traversal_obs = torch.tensor(env.traverse())
    data = [traversal_obs]
    repeat_num = math.ceil(cfg.dataset_size / traversal_obs.size(0))

    for _ in range(repeat_num-1):
        traversal_obs = torch.tensor(env.traverse())
        data.append(traversal_obs)

    return torch.cat(data, dim=0)


def collect_data(cfg):
    """collect data from observations of virtual env
    :param cfg: instance of CollectDatasetConfiguration
    """
    # use different function according to env type
    if cfg.type == "gym":
        return collect_env_data_gym(cfg)
    elif cfg.type == "custom":
        return collect_env_data_custom(cfg)
    else:
        raise NotImplementedError