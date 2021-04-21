import os
import logging
import argparse

import torch

from models.rl import *
from envs.wrapped_envs import make_vec_envs
from load_config import load_config_from_files
from utils.exp import get_seed


def parse_args():
    """parse command line arguments"""
    desc = "MBPO tests (rendering)"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--config-file', nargs='+', type=str,
                        help='experiemnt configuration file path(s) (yaml format supported now)')
    parser.add_argument('--num-episode', type=int, metavar='N',
                        help='number of episodes take during testing')

    args = parser.parse_args()

    return args


def load_agent(actor, save_dir):
    state_dicts_path = os.path.join(save_dir, 'state_dicts.pt')
    state_dicts = torch.load(state_dicts_path)

    actor.load_state_dict(state_dicts['actor'])


def main(args):
    """main procedure"""
    try:
        cfgs = load_config_from_files(args.config_file)
        rl_cfg = cfgs['rl']
        env_cfg = rl_cfg.env
        algos_cfg = rl_cfg.algos
        sac_cfg = algos_cfg['sac']
        num_ep = args.num_episode

        # results tmp dir
        log_dir = os.path.join('results', env_cfg.env_name, 'tmp', 'log')

        # load model
        save_model_cfg = rl_cfg.save_model_config
        save_dir = save_model_cfg.default_dir or './save'
        save_dir = os.path.join(save_dir, save_model_cfg.path)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # create environment
        device = torch.device('cpu')
        eval_envs = make_vec_envs(env_cfg.env_name, get_seed(),
                                  env_cfg.num, env_cfg.gamma, log_dir,
                                  device, allow_early_resets=True,
                                  norm_reward=False, norm_obs=False)

        state_dim = eval_envs.observation_space.shape[0]
        action_space = eval_envs.action_space

        actor = Actor(state_dim, action_space, sac_cfg['actor_hidden_dims'],
                      state_normalizer=None, use_limited_entropy=False,
                      use_tanh_squash=True, use_state_dependent_std=True)
        load_agent(actor, save_dir)

        eval_episode_rewards = []
        eval_episode_lengths = []

        states = eval_envs.reset()
        while len(eval_episode_rewards) < num_ep:
            eval_envs.render()

            with torch.no_grad():
                actions = actor.act(states, deterministic=True)['actions']
            states, _, _, infos = eval_envs.step(actions)

            eval_episode_rewards.extend([info['episode']['r'] for info in infos if 'episode' in info])
            eval_episode_lengths.extend([info['episode']['l'] for info in infos if 'episode' in info])

        eval_envs.close()

        print('rewards: \n{}'.format(eval_episode_rewards))
        print('lengths: \n{}'.format(eval_episode_lengths))

    except Exception as e:
        logging.error("MBPO testing encountered error: \n{}".format(e), exc_info=True)
    else:
        logging.info("MBPO testing done")


if __name__ == "__main__":
    args = parse_args()

    main(args)
