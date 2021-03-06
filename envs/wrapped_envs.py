from __future__ import annotations
import os
from typing import Optional, TYPE_CHECKING

from gym.wrappers import TimeLimit
import torch

from .virtual_env import VecVirtualEnv, FakeEnv
from .benchmarking_envs.benchmarking_envs import make_benchmarking_env, make_atari_env
from thirdparty.base_vec_env import VecEnvWrapper
from thirdparty.dummy_vec_env import DummyVecEnv
from thirdparty.subproc_vec_env import SubprocVecEnv
from thirdparty.vec_normalize import VecNormalize
from thirdparty.monitor import Monitor

if TYPE_CHECKING:
    from models.rl.dynamics import BaseDynamics
    from models.rl.term_fn import TerminationFn


def make_env(env_id, seed, rank, log_dir, allow_early_resets, max_episode_steps):
    def _thunk():
        env = make_benchmarking_env(env_id)
        env = TimeLimit(env, max_episode_steps)

        env.seed(seed + rank)
        log_dir_ = os.path.join(log_dir, str(rank)) if log_dir is not None else log_dir
        env = Monitor(env, log_dir_, allow_early_resets=allow_early_resets)

        return env

    return _thunk


def make_env_atari(env_id, seed, rank, log_dir, allow_early_resets):
    """return the function to make a new atari env"""
    def _thunk():
        env = make_atari_env(env_id)

        env.seed(seed + rank)
        new_log_dir = os.path.join(log_dir, str(rank)) if log_dir is not None else log_dir
        if new_log_dir and not os.path.exists(new_log_dir):
            os.makedirs(new_log_dir)
        env = Monitor(env, new_log_dir, allow_early_resets)

        return env

    return _thunk


def make_vec_envs(env_name: str,
                  seed: int,
                  num_envs: int,
                  gamma: Optional[float],
                  log_dir: Optional[str],
                  device: torch.device,
                  allow_early_resets: bool,
                  max_episode_steps: int = 1000,
                  norm_reward=True,
                  norm_obs=True,
                  ):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, max_episode_steps)
        for i in range(num_envs)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False, norm_obs=norm_obs)
        else:
            envs = VecNormalize(envs, gamma=gamma, norm_reward=norm_reward, norm_obs=norm_obs)

    envs = VecPyTorch(envs, device)

    return envs


def make_vec_atari_envs(env_name: str,
                        seed: int,
                        num_envs: int,
                        gamma: Optional[int],
                        log_dir: Optional[str],
                        device: torch.device,
                        allow_early_resets: bool,
                        norm_reward=True,
                        ):
    """make vectorized environments"""
    num_envs = num_envs or 1
    envs = [
        make_env_atari(env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_envs)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False, norm_obs=False)
        else:
            envs = VecNormalize(envs, gamma=gamma, norm_reward=norm_reward, norm_obs=False)

    envs = VecPyTorch(envs, device)

    return envs


def make_vec_virtual_envs(env_name: str,
                          dynamics: BaseDynamics,
                          seed: int,
                          num_envs: int,
                          gamma: Optional[float],
                          device: torch.device,
                          norm_reward=False,
                          norm_obs=False,
                          **kwargs
                          ):
    envs = VecVirtualEnv(dynamics, make_benchmarking_env(env_name), num_envs, seed, **kwargs)

    if (len(envs.observation_space.shape) == 1 and norm_obs) or norm_reward:
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False, norm_obs=norm_obs)
        else:
            envs = VecNormalize(envs, gamma=gamma, norm_reward=norm_reward, norm_obs=norm_obs)

    envs = VecPyTorch(envs, device)

    return envs


def make_fake_env(dynamics: BaseDynamics, term_fn: TerminationFn):
    """return fake env which only do transition data generation"""
    return FakeEnv(dynamics, term_fn)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_with_states(self, states: torch.Tensor, actions: torch.Tensor):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        return self.venv.step_with_states(states, actions)

    def step_async(self, actions: torch.Tensor):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        new_method_args = []
        new_method_kwargs = {}
        for method_arg in method_args:
            if type(method_arg) == torch.Tensor:
                new_method_args.append(method_arg.cpu().numpy())
        for method_arg_k, method_arg_v in method_kwargs.items():
            if type(method_arg_v) == torch.Tensor:
                new_method_kwargs[method_arg_k] = method_arg_v.cpu().numpy()
        self.venv.env_method(method_name, *new_method_args, indices, **new_method_kwargs)


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


