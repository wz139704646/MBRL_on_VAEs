# execute experiment about all kinds of vaes training and testing
import os
import time
import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import deque

from algos import *
from models.rl import *
from models.vaes import *
from models.normalizer import RunningNormalizer
from base_experiment import BaseExperiment
from handlers.configure_log import adapt_configure_log
from handlers.rl_evaluate import evaluate
from envs.wrapped_envs import make_vec_envs, make_vec_virtual_envs
from storages import SimpleUniversalBuffer as Buffer, MixtureBuffer
from utils.exp import plot_losses, set_seed, get_seed, log_and_write


class MBPOExperiment(BaseExperiment):
    """VAE training and testing (basic) experiment"""

    def apply_configs(self):
        """apply the configurations"""
        try:
            # environment setting
            if "general" in self.exp_configs:
                set_seed(self.exp_configs['general'].seed)

            # ensure directories
            rl_cfg = self.exp_configs['rl']
            save_model_cfg = rl_cfg.save_model_config
            save_res_cfg = rl_cfg.save_result_config
            current_time = datetime.datetime.now().strftime('%b%d_%H%M%S')

            # ensure model saving directory
            self.save_dir = save_model_cfg.default_dir or './save'
            # path setting for rl exp is actually directory path
            self.save_dir = os.path.join(self.save_dir, save_model_cfg.path)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            # ensure test results directory
            results_dir = save_res_cfg.default_dir or './results'
            self.log_dir = os.path.join(results_dir, current_time, 'log')
            self.eval_log_dir = os.path.join(results_dir, current_time, 'log_eval')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.eval_log_dir):
                os.makedirs(self.eval_log_dir)

            # other settings
            self.device = torch.device(rl_cfg.device)

            log_config = self.exp_configs["log"] if "log" in self.exp_configs else None
            self.log_comp = adapt_configure_log(log_config)
        except Exception:
            raise

    def _load(self):
        """load the trained models and buffers"""
        rl_cfg = self.exp_configs['rl']

        if rl_cfg.model_load_path:
            state_dicts = torch.load(rl_cfg.model_load_path)
            dynamics_sd, actor_sd, q_critic1_sd, q_critic2_sd, q_critic_target1_sd, q_critic_target2_sd = \
                itemgetter('dynamics', 'actor', 'q_critic1', 'q_critic2',
                           'q_critic_target1', 'q_critic_target2')(state_dicts)
            self.model.dynamics.load_state_dict(dynamics_sd)
            self.agent.actor.load_state_dict(actor_sd)
            self.agent.q_critic1.load_state_dict(q_critic1_sd)
            self.agent.q_critic2.load_state_dict(q_critic2_sd)
            self.agent.target_q_critic1.load_state_dict(q_critic_target1_sd)
            self.agent.target_q_critic2.load_state_dict(q_critic_target2_sd)
        if rl_cfg.buffer_load_path:
            # virtual buffer does not need loading
            self.real_buffer.load(rl_cfg.buffer_load_path)

    def warm_up(self):
        """generate warm up samples"""
        rl_cfg = self.exp_configs['rl']
        env_cfg = rl_cfg.env
        mbpo_cfg = rl_cfg.algos['mbpo']

        for _ in range(mbpo_cfg['num_warmup_samples']):
            real_actions = torch.tensor(
                [self.real_envs.action_space.sample() for _ in range(env_cfg.num)]
            ).to(self.device)
            real_next_states, real_rewards, real_dones, real_infos = self.real_envs.step(real_actions)
            real_masks = torch.tensor([[0.0] if done else [1.0] for done in real_dones], dtype=torch.float32)
            self.real_buffer.insert(states=self.real_states, actions=real_actions, rewards=real_rewards,
                                    masks=real_masks, next_states=real_next_states)
            self.real_states = real_next_states

            self.real_episode_rewards.extend([info['episode']['r'] for info in real_infos if 'episode' in info])
            self.real_episode_lengths.extend([info['episode']['l'] for info in real_infos if 'episode' in info])

        recent_states, recent_actions = itemgetter('states', 'actions') \
            (self.real_buffer.get_recent_samples(mbpo_cfg['num_warmup_samples'] - mbpo_cfg['model_update_interval']))
        self.state_normalizer.update(recent_states)
        self.action_normalizer.update(recent_actions)

    def before_run(self, **kwargs):
        """preparations needed be done before run the experiment"""
        try:
            rl_cfg = self.exp_configs["rl"]
            env_cfg = rl_cfg.env
            algos_cfg = rl_cfg.algos
            mbpo_cfg = algos_cfg['mbpo']
            sac_cfg = algos_cfg['sac']

            # init real envs
            self.real_envs = make_vec_envs(env_cfg.env_name, get_seed(), env_cfg.num,
                                           env_cfg.gamma, self.log_dir,
                                           self.device, allow_early_resets=True,
                                           norm_reward=False, norm_obs=False)

            # extract dim info
            state_dim = self.real_envs.observation_space.shape[0]
            action_space = self.real_envs.action_space
            action_dim = self.real_envs.action_space.shape[0]
            datatype = {'states': {'dims': [state_dim]}, 'next_states': {'dims': [state_dim]},
                        'actions': {'dims': [action_dim]}, 'rewards': {'dims': [1]}, 'masks': {'dims': [1]}}

            # init normalizers
            self.state_normalizer = RunningNormalizer(state_dim)
            self.action_normalizer = RunningNormalizer(action_dim)
            self.state_normalizer.to(self.device)
            self.action_normalizer.to(self.device)

            # init env dynamics
            dynamics = EnsembleRDynamics(state_dim, action_dim, 1,
                                         mbpo_cfg['dynamics_hidden_dims'],
                                         mbpo_cfg['num_dynamics_networks'],
                                         mbpo_cfg['num_elite_dynamics_networks'],
                                         self.state_normalizer, self.action_normalizer)
            dynamics.to(self.device)

            # init sac agent (model-free algo)
            actor = Actor(state_dim, action_space, sac_cfg['actor_hidden_dims'],
                          state_normalizer=None, use_limited_entropy=False,
                          use_tanh_squash=True, use_state_dependent_std=True)
            actor.to(self.device)

            q_critic1 = QCritic(state_dim, action_space, sac_cfg['critic_hidden_dims'])
            q_critic2 = QCritic(state_dim, action_space, sac_cfg['critic_hidden_dims'])
            q_critic_target1 = QCritic(state_dim, action_space, sac_cfg['critic_hidden_dims'])
            q_critic_target2 = QCritic(state_dim, action_space, sac_cfg['critic_hidden_dims'])
            q_critic1.to(self.device)
            q_critic2.to(self.device)
            q_critic_target1.to(self.device)
            q_critic_target2.to(self.device)
            target_entropy = sac_cfg['target_entropy'] or \
                -np.prod(self.real_envs.action_space.shape).item()
            self.agent = SAC(actor, q_critic1, q_critic2, q_critic_target1, q_critic_target2,
                        sac_cfg['batch_size'], sac_cfg['num_grad_steps'], env_cfg.gamma,
                        1.0, sac_cfg['actor_lr'], sac_cfg['critic_lr'], sac_cfg['soft_target_tau'],
                        target_entropy=target_entropy)

            # init virtual env
            self.virtual_envs = make_vec_virtual_envs(env_cfg.env_name, dynamics, get_seed(), 0,
                                                 env_cfg.gamma, self.device, use_predicted_reward=True)

            # init virtual env buffer
            self.base_virtual_buffer_sz = mbpo_cfg['rollout_batch_size'] * env_cfg.max_episode_steps * \
                mbpo_cfg['num_model_retain_epochs'] // mbpo_cfg['model_update_interval']
            self.virtual_buffer = Buffer(self.base_virtual_buffer_sz, datatype)
            self.virtual_buffer.to(self.device)
            self.agent.check_buffer(self.virtual_buffer)

            # init mbpo model
            self.model = MBPO(dynamics, mbpo_cfg['dynamics_batch_size'],
                              rollout_schedule=mbpo_cfg['rollout_schedule'],
                              verbose=1, lr=mbpo_cfg['lr'],
                              l2_loss_coefs=mbpo_cfg['l2_loss_coefs'],
                              max_num_epochs=mbpo_cfg['max_num_epochs'],
                              logger=self.log_comp['logger'])

            # init real env buffer
            self.real_buffer = Buffer(mbpo_cfg['real_buffer_size'], datatype)
            self.real_buffer.to(self.device)
            self.model.check_buffer(self.real_buffer)

            # init policy buffer
            if mbpo_cfg['real_sample_ratio'] > 0:
                ratio = mbpo_cfg['real_sample_ratio']
                self.policy_buffer = MixtureBuffer([self.virtual_buffer, self.real_buffer], [1 - ratio, ratio])
            else:
                self.policy_buffer = self.virtual_buffer

            # load checkpoints
            if ((not rl_cfg.model_load_path) ^ (not rl_cfg.buffer_load_path)):
                raise Exception("partial loading model or buffer may cause error")
            self._load()

            # init obs, reward and length
            self.real_states = self.real_envs.reset()
            self.real_episode_rewards = deque(maxlen=30)
            self.real_episode_lengths = deque(maxlen=30)
            self.warm_up()

            # log and summary
            logger = self.log_comp['logger']
            writer = self.log_comp['summary_writer'] if 'summary_writer' in self.log_comp else None

            logger.info('Env Observation Space Shape: {}'.format(self.real_envs.observation_space.shape))
            logger.info('Env Action Space shape: {}'.format(self.real_envs.action_space.shape))
            logger.info('Input States Shape: {}'.format(self.real_states.shape))
            logger.info('Dynamics Model: \n{}'.format(dynamics))
            logger.info('Actor Model: \n{}'.format(actor))
            logger.info('Critic Model: \n')
            logger.info('1: \n{}'.format(q_critic1))
            logger.info('2: \n{}'.format(q_critic2))
            logger.info('target 1: \n{}'.format(q_critic_target1))
            logger.info('target 2: \n{}'.format(q_critic_target2))
            # fake_input = (torch.zeros(self.real_envs.observation_space.shape),
            #               torch.zeros(self.real_envs.action_space.shape))
            # writer.add_graph(dynamics, fake_input)
            # writer.add_graph(actor, torch.zeros(self.real_envs.observation_space.shape))
            # writer.add_graph(q_critic1, fake_input)
            # writer.add_graph(q_critic2, fake_input)
            # writer.add_graph(q_critic_target1, fake_input)
            # writer.add_graph(q_critic_target2, fake_input)
        except Exception:
            raise

    def run(self, **kwargs):
        """run the main part of the experiment"""
        try:
            rl_cfg = self.exp_configs["rl"]
            env_cfg = rl_cfg.env
            algos_cfg = rl_cfg.algos
            mbpo_cfg = algos_cfg['mbpo']

            logger = self.log_comp['logger']
            writer = self.log_comp['summary_writer'] if 'summary_writer' in self.log_comp else None
            start = time.time()

            for epoch in range(mbpo_cfg['num_total_epochs']):
                logger.info('Epoch {}:'.format(epoch))

                # update rollout length k
                self.model.update_rollout_length(epoch)

                for i in range(env_cfg.max_episode_steps):
                    self.losses = {}
                    if i % mbpo_cfg['model_update_interval'] == 0:
                        # update predictive model
                        recent_states, recent_actions = itemgetter('states', 'actions') \
                            (self.real_buffer.get_recent_samples(mbpo_cfg['model_update_interval']))
                        self.state_normalizer.update(recent_states)
                        self.action_normalizer.update(recent_actions)

                        self.losses.update(self.model.update(self.real_buffer))

                        # generate data in virtual env
                        initial_states = next(self.real_buffer.get_batch_generator_inf(
                            mbpo_cfg['rollout_batch_size']))['states']
                        new_virtual_buffer_sz = self.base_virtual_buffer_sz * self.model.num_rollout_steps
                        self.virtual_buffer.resize(new_virtual_buffer_sz)
                        self.model.generate_data(self.virtual_envs, self.virtual_buffer,
                                                 initial_states, self.agent.actor)

                    # take actions in real env and add to buffer
                    with torch.no_grad():
                        real_actions = self.agent.actor.act(self.real_states)['actions']

                    real_next_states, real_rewards, real_dones, real_infos = self.real_envs.step(real_actions)
                    real_masks = torch.tensor([[0.0] if done else [1.0] for done in real_dones], dtype=torch.float32)
                    self.real_buffer.insert(states=self.real_states, actions=real_actions, rewards=real_rewards,
                                            masks=real_masks, next_states=real_next_states)
                    self.real_states = real_next_states
                    self.real_episode_rewards.extend(
                        [info['episode']['r'] for info in real_infos if 'episode' in info])
                    self.real_episode_lengths.extend(
                        [info['episode']['l'] for info in real_infos if 'episode' in info])

                    # update policy
                    self.losses.update(self.agent.update(self.policy_buffer))

                    if i % rl_cfg.log_interval == 0:
                        # log intermediate informations
                        # keys with '/' will be recorded in the tensorboard
                        time_elapsed = time.time() - start
                        num_env_steps = epoch * env_cfg.max_episode_steps + i + mbpo_cfg['num_warmup_samples']
                        log_infos = {'/time_elapsed': time_elapsed, 'samples_collected': num_env_steps}

                        if len(self.real_episode_rewards) > 0:
                            log_infos['perf/ep_rew_real'] = np.mean(self.real_episode_rewards)
                            log_infos['perf/ep_len_real'] = np.mean(self.real_episode_lengths)
                        for k in self.losses.keys():
                            log_infos['loss/' + k] = self.losses[k]
                        log_and_write(log_infos, global_step=num_env_steps, logger=logger, writer=writer)

                if (epoch + 1) % rl_cfg.eval_interval == 0:
                    episode_rewards_real_eval, episode_lengths_real_eval = \
                        evaluate(self.agent.actor, env_cfg.env_name, get_seed(), 10, self.eval_log_dir,
                                 self.device, norm_reward=False, norm_obs=False)
                    log_infos = {'perf/ep_rew_real_eval': np.mean(episode_rewards_real_eval),
                                 'perf/ep_len_real_eval': np.mean(episode_lengths_real_eval)}
                    log_and_write(log_infos, logger=logger, writer=writer,
                                  global_step=(epoch + 1) * env_cfg.max_episode_steps + mbpo_cfg['num_warmup_samples'])

                if (epoch + 1) % rl_cfg.save_interval == 0:
                    self._save()
        except Exception:
            raise

    def _save(self, extra=None):
        """save the model and configurations"""
        rl_cfg = self.exp_configs['rl']

        save_model_cfg = rl_cfg.save_model_config
        state_dicts = {'dynamics': self.model.dynamics.state_dict(),
                       'actor': self.agent.actor.state_dict(),
                       'q_critic1': self.agent.q_critic1.state_dict(),
                       'q_critic2': self.agent.q_critic2.state_dict(),
                       'q_critic_target1': self.agent.target_q_critic1.state_dict(),
                       'q_critic_target2': self.agent.target_q_critic2.state_dict()}
        # store the cfgs into the state dict pack for convenience
        cfgs = {}
        for cfg_key in save_model_cfg.store_cfgs:
            cfgs[cfg_key] = self.exp_configs[cfg_key].raw
        if cfgs:
            state_dicts['exp_configs'] = cfgs

        torch.save(state_dicts, self.save_dir + '/state_dicts.pt')
        self.real_buffer.save(self.save_dir + '/real_buffer.pt')

    def after_run(self, **kwargs):
        """cleaning up needed be done after run the experiment"""
        # save model, plot losses
        try:
            self.real_envs.close()
            self.virtual_envs.close()
            if "summary_writer" in self.log_comp:
                writer = self.log_comp["summary_writer"]
                writer.close()
        except Exception:
            raise
