import torch
from gym.spaces import Discrete

from envs.wrapped_envs import make_vec_envs, get_vec_normalize, make_vec_atari_envs
from utils.exp import unwrap_onehot_to_discrete


def evaluate(actor, env_name, seed, num_episode, eval_log_dir,
             device, norm_reward=False, norm_obs=True, obs_rms=None):
    """evaluate the rl training result"""
    eval_envs = make_vec_envs(env_name, seed, 1, None, eval_log_dir, device, allow_early_resets=True,
                              norm_obs=norm_obs, norm_reward=norm_reward)
    vec_norm = get_vec_normalize(eval_envs)

    if vec_norm is not None and norm_obs:
        assert obs_rms is not None
        vec_norm.training = False
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []
    eval_episode_lengths = []

    states = eval_envs.reset()
    while len(eval_episode_rewards) < num_episode:
        with torch.no_grad():
            actions = actor.act(states, deterministic=True)['actions']

        states, _, _, infos = eval_envs.step(actions)

        eval_episode_rewards.extend([info['episode']['r'] for info in infos if 'episode' in info])
        eval_episode_lengths.extend([info['episode']['l'] for info in infos if 'episode' in info])

    eval_envs.close()

    return eval_episode_rewards, eval_episode_lengths


def evaluate_atari(actor, env_name, seed, num_episode, eval_log_dir, device,
                   norm_reward=False, encoder=None, encode_fn=None):
    """evaluate the rl training result in evaluation atari env"""
    def wrap_obs_state(obs):
        if encoder:
            # encode obseration to state
            with torch.no_grad():
                wrapped_st = encode_fn(encoder, obs)
        else:
            # use obs as state
            wrapped_st = torch.flatten(obs).view(obs.size(0), -1)

        return wrapped_st

    eval_envs = make_vec_atari_envs(
        env_name, seed, 1, None, eval_log_dir, device,
        allow_early_resets=True, norm_reward=norm_reward)
    action_space = eval_envs.action_space

    eval_episode_rewards = []
    eval_episode_lengths = []

    obs = eval_envs.reset()
    states = wrap_obs_state(obs)
    while len(eval_episode_rewards) < num_episode:
        with torch.no_grad():
            actions = actor.act(states, deterministic=True)['actions']

        if isinstance(action_space, Discrete):
            # unwrap onehot action
            actions = torch.tensor(unwrap_onehot_to_discrete(actions))

        states, _, _, infos = eval_envs.step(actions)
        states = wrap_obs_state(obs)

        eval_episode_rewards.extend([info['episode']['r'] for info in infos if 'episode' in info])
        eval_episode_lengths.extend([info['episode']['l'] for info in infos if 'episode' in info])

    eval_envs.close()

    return eval_episode_rewards, eval_episode_lengths


