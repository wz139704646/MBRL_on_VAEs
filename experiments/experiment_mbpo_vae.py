# execute experiment about all kinds of vaes training and testing
import os
import time
import datetime
import torch
import torch.autograd
import numpy as np
from operator import itemgetter
from collections import deque, Iterable

from algos import *
from models.rl import *
from models.vaes import *
from models.normalizer import RunningNormalizer
from base_experiment import BaseExperiment
from handlers.configure_log import adapt_configure_log
from handlers.rl_evaluate import evaluate_atari
from handlers.load_model import adapt_load_model
from handlers.load_data import adapt_get_data_gen_with_buffer
from handlers.configure_optimizer import adapt_configure_optimizer
from handlers.train_test import adapt_kv_generator_train, adapt_kv_generator_test
from handlers.save_model import get_extra_setting
from envs.wrapped_envs import make_vec_atari_envs, make_fake_env
from storages import SimpleUniversalBuffer as Buffer, MixtureBuffer
from utils.exp import set_seed, get_seed, log_and_write, wrap_discrete_to_onehot, unwrap_onehot_to_discrete


def vae_encode(vae_model, x, device=None):
    """encode function for vae model"""
    return vae_model.reparameterize(*vae_model.encode(x)).to(device or x.device)


def split_buffer(buffer, division):
    """split the buffer according to division"""
    full_indices = np.arange(buffer.size)
    np.random.shuffle(full_indices)

    if division is None:
        # return the entire buffer
        return full_indices

    start = 0
    indices = []
    for d in division:
        end = start + int(buffer.size * d)
        indices.append(full_indices[start:end])
        start = end

    return indices


class MBPOVAEsExperiment(BaseExperiment):
    """VAE training and testing (basic) experiment"""

    def apply_configs(self):
        """apply the configurations"""
        try:
            # environment setting
            self.debug = False
            if "general" in self.exp_configs:
                general_cfg = self.exp_configs['general']
                set_seed(general_cfg.seed)
                self.debug = general_cfg.debug

            # debug setting
            torch.autograd.set_detect_anomaly(self.debug)

            # ensure directories
            rl_cfg = self.exp_configs['rl']
            encoding_cfg = rl_cfg.encoding_config
            save_model_cfg = rl_cfg.save_model_config
            save_res_cfg = rl_cfg.save_result_config
            current_time = datetime.datetime.now().strftime('%b%d_%H%M%S')
            self.encoding = encoding_cfg is not None

            # ensure model saving directory
            self.save_dir = save_model_cfg.default_dir or './save'
            # path setting for rl exp is actually directory path
            self.save_dir = os.path.join(self.save_dir, save_model_cfg.path)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            # encoding model part
            if self.encoding:
                train_config = encoding_cfg.train_config
                model_config = encoding_cfg.model_config
                if train_config.save_config or model_config.path:
                    model_file = train_config.save_config.default_dir
                if train_config.save_config.path:
                    model_file = os.path.join(model_file, train_config.save_config.path)
                elif model_config.path:
                    # refer to model config
                    model_file = os.path.join(model_config.default_dir, model_config.path)
                else:
                    # use timestamp as name
                    model_file = model_file if model_file != '' else model_config.default_dir
                    model_file = os.path.join(model_file, "{}.pth.tar".format(int(time.time())))
                if not os.path.exists(os.path.dirname(model_file)):
                    os.makedirs(os.path.dirname(model_file))
                self.encoding_model_file = model_file

            # ensure test results directory
            results_dir = save_res_cfg.default_dir or './results'
            self.log_dir = os.path.join(results_dir, current_time, 'log')
            self.eval_log_dir = os.path.join(results_dir, current_time, 'log_eval')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.eval_log_dir):
                os.makedirs(self.eval_log_dir)
            # encoding test results directory making
            if self.encoding:
                test_config = encoding_cfg.test_config
                encoding_results_dir = os.path.join(results_dir, current_time, 'encoding_res')
                if not os.path.exists(encoding_results_dir):
                    os.makedirs(encoding_results_dir)
                self.encoding_results_dir = encoding_results_dir
                test_config.save_config.default_dir = encoding_results_dir

            # ensure monitoring logs directory
            monitor_dir = rl_cfg.monitor_dir
            self.monitor_dir = os.path.join(monitor_dir, current_time)
            if not os.path.exists(self.monitor_dir):
                os.makedirs(self.monitor_dir)

            # other settings
            self.device = torch.device(rl_cfg.device)
            if self.encoding:
                self.exp_configs['rl'].encoding_config.train_config.extra["device"] = self.device
                self.exp_configs['rl'].encoding_config.test_config.extra["device"] = self.device

            log_config = self.exp_configs["log"] if "log" in self.exp_configs else None
            self.log_comp = adapt_configure_log(log_config)

            logger = self.log_comp['logger']
            for cfg_k in self.exp_configs.keys():
                logger.debug('Experiment Config [{}]: \n{}'.format(cfg_k, self.exp_configs))
        except Exception:
            raise

    def _load(self):
        """load the trained models, buffers and extra info"""
        rl_cfg = self.exp_configs['rl']

        if rl_cfg.model_load_path:
            state_dicts = torch.load(rl_cfg.model_load_path)
            dynamics_sd, actor_sd, q_critic1_sd, q_critic2_sd, q_critic_target1_sd, q_critic_target2_sd, term_fn_sd = \
                itemgetter('dynamics', 'actor', 'q_critic1', 'q_critic2',
                           'q_critic_target1', 'q_critic_target2', 'term_fn')(state_dicts)
            self.model.dynamics.load_state_dict(dynamics_sd)
            self.model.term_fn.load_state_dict(term_fn_sd)
            self.agent.actor.load_state_dict(actor_sd)
            self.agent.q_critic1.load_state_dict(q_critic1_sd)
            self.agent.q_critic2.load_state_dict(q_critic2_sd)
            self.agent.target_q_critic1.load_state_dict(q_critic_target1_sd)
            self.agent.target_q_critic2.load_state_dict(q_critic_target2_sd)
        if rl_cfg.buffer_load_path:
            # virtual buffer does not need loading
            if self.encoding:
                self.obs_buffer.load(rl_cfg.buffer_load_path)
            else:
                self.real_buffer.load(rl_cfg.buffer_load_path)
        if self.encoding and rl_cfg.encoding_load_path:
            # load encoding model (vae)
            load_res = adapt_load_model(rl_cfg.encoding_load_path)
            self.encoding_model = load_res['model']
        if rl_cfg.extra_load_path:
            # load extra info (e.g. epoch)
            extra = torch.load(rl_cfg.extra_load_path)
            self.epoch = extra['epoch']
            if 'encoding_global_step' in extra:
                self.encoding_global_step = extra['encoding_global_step']

    def encode(self, obs):
        """encode obs into states"""
        return vae_encode(self.encoding_model, obs, self.device)

    def set_action_space(self, action_space):
        """set env action space related properties"""
        self.action_space = action_space

        if len(action_space.shape) == 0:
            # discrete action space, use one-hot action
            n = action_space.n
            self.action_shape = (n, )
            self.action_dim = n
        else:
            self.action_shape = action_space.shape
            self.action_dim = self.action_shape[0]

    def wrap_action(self, action):
        """wrap the action into appropriate form according to action space"""
        if len(self.action_space.shape) == 0:
            # discrete action space, use one-hot action
            return wrap_discrete_to_onehot(action, self.action_shape)
        else:
            return action

    def unwrap_action(self, wrapped_action):
        """tansform the action to the raw to use in the env"""
        if len(self.action_space.shape) == 0:
            # convert to discrete action
            return torch.tensor(unwrap_onehot_to_discrete(wrapped_action.cpu()))
        else:
            return wrapped_action

    def warm_up(self):
        """generate warm up samples"""
        rl_cfg = self.exp_configs['rl']
        env_cfg = rl_cfg.env
        mbpo_cfg = rl_cfg.algos['mbpo']

        for _ in range(mbpo_cfg['num_warmup_samples']):
            # generate warming up samples
            real_actions = torch.tensor(
                [self.wrap_action(self.real_envs.action_space.sample()) for _ in range(env_cfg.num)]
            ).to(self.device)
            real_next_obs, real_rewards, real_dones, real_infos = self.real_envs.step(
                self.unwrap_action(real_actions))
            real_masks = torch.tensor([[0.0] if done else [1.0] for done in real_dones], dtype=torch.float32)
            if self.encoding:
                self.obs_buffer.insert(obs=self.real_obs, actions=real_actions, rewards=real_rewards,
                                       masks=real_masks, next_obs=real_next_obs)
            else:
                real_states = torch.flatten(self.real_obs).view(self.real_obs.size(0), -1)
                real_next_states = torch.flatten(real_next_obs).view(real_next_obs.size(0), -1)
                self.real_buffer.insert(states=real_states, actions=real_actions, rewards=real_rewards,
                                        masks=real_masks, next_states=real_next_states)
            self.real_obs = real_next_obs

            self.real_episode_rewards.extend([info['episode']['r'] for info in real_infos if 'episode' in info])
            self.real_episode_lengths.extend([info['episode']['l'] for info in real_infos if 'episode' in info])

    def update_encoding_model(self):
        """update encoding model via observation buffer data"""
        rl_cfg = self.exp_configs['rl']
        encoding_cfg = rl_cfg.encoding_config
        model_cfg = encoding_cfg.model_config
        train_cfg = encoding_cfg.train_config
        test_cfg = encoding_cfg.test_config
        writer = None

        if "summary_writer" in self.log_comp:
            writer = self.log_comp["summary_writer"]

        epochs = train_cfg.epochs
        batch_size = encoding_cfg.batch_size
        indices = split_buffer(self.obs_buffer, encoding_cfg.division)
        max_update_steps = encoding_cfg.max_update_steps # max updates during a epoch
        max_test_steps = encoding_cfg.max_test_steps # max tests during a epoch
        num_batches_train = min(max_update_steps, len(indices[0]) // batch_size)
        num_batches_test = min(max_test_steps, len(indices[1]) // batch_size)
        losses = {}

        for epoch in range(epochs):
            # train one epoch
            train_gen, test_gen = adapt_get_data_gen_with_buffer(
                model_cfg.model_name, self.obs_buffer, batch_size, indices, gen_type='epoch')
            avg_loss = adapt_kv_generator_train(
                model_cfg.model_name, self.encoding_model,
                self.encoding_optimizer, train_gen, 'obs',
                num_batches_train, batch_size, train_cfg,
                self.encoding_global_step, self.log_comp)
            # store all kinds of losses
            for k in avg_loss.keys():
                if k in losses:
                    losses[k].append(avg_loss[k])
                else:
                    losses[k] = [avg_loss[k]]

                # tensorboard logging
                if writer is not None:
                    writer.add_scalar("loss/train " + k, avg_loss[k], self.encoding_global_step)

            # save the last epoch results
            save_res = epoch == epochs - 1
            test_loss = adapt_kv_generator_test(
                model_cfg.model_name, self.encoding_model,
                test_gen, 'obs', num_batches_test, batch_size,
                train_cfg, test_cfg, self.encoding_global_step,
                self.log_comp, save=save_res)
            # tensorboard logging
            if writer is not None:
                for k in test_loss.keys():
                    writer.add_scalar("loss/test " + k, test_loss[k], self.encoding_global_step)

            self.encoding_model.update_epoch()
            self.encoding_global_step += 1

    def encode_obs_buffer(self):
        """encode the obs buffer to update state buffer"""
        recent_obs, recent_actions, recent_rewards, recent_masks, recent_next_obs = \
            itemgetter(
                'obs', 'actions', 'rewards', 'masks', 'next_obs'
            )(self.obs_buffer.get_recent_samples(self.obs_buffer.size))
        with torch.no_grad():
            # encode obs
            recent_states, recent_next_states = self.encode(recent_obs), self.encode(recent_next_obs)
        self.real_buffer.insert(states=recent_states, actions=recent_actions, rewards=recent_rewards,
                                masks=recent_masks, next_states=recent_next_states)

    def get_encode_update_interval(self, epoch):
        """get the encoder update interval (steps) from config and current epoch"""
        rl_cfg = self.exp_configs['rl']
        algos_cfg = rl_cfg.algos
        mbpo_cfg = algos_cfg['mbpo']

        encoder_interval = mbpo_cfg['encoder_update_interval']

        if isinstance(encoder_interval, Iterable):
            # schedule the update interval
            min_ep, max_ep, st_steps, en_steps = encoder_interval
            if epoch <= min_ep:
                interval = st_steps
            else:
                delta = (epoch - min_ep) / (max_ep - min_ep)
                delta = min(delta, 1)
                interval = st_steps + (en_steps - st_steps) * delta
        else:
            # fixed interval
            interval = encoder_interval

        return interval

    def before_run(self, **kwargs):
        """preparations needed be done before run the experiment"""
        try:
            rl_cfg = self.exp_configs["rl"]
            env_cfg = rl_cfg.env
            algos_cfg = rl_cfg.algos
            mbpo_cfg = algos_cfg['mbpo']
            sac_cfg = algos_cfg['sac']
            if self.encoding:
                encoding_cfg = rl_cfg.encoding_config
                encoding_model_cfg = encoding_cfg.model_config
                encoding_train_cfg = encoding_cfg.train_config

            # init real env
            self.real_envs = make_vec_atari_envs(
                env_cfg.env_name, get_seed(), env_cfg.num, env_cfg.gamma, self.log_dir,
                self.device, allow_early_resets=True, norm_reward=False)
            action_space = self.real_envs.action_space
            self.set_action_space(action_space)

            # init encoding model, buffer and state dimension
            if self.encoding:
                model_name = encoding_model_cfg.model_name
                model_args = encoding_model_cfg.model_args
                if "input_size" not in model_args:
                    model_args["input_size"] = self.real_envs.observation_space.shape
                self.encoding_model = eval(model_name)(**model_args)
                self.encoding_model.to(self.device)
                self.encoding_optimizer = adapt_configure_optimizer(
                    encoding_model_cfg.model_name, self.encoding_model, encoding_train_cfg.optimizer_config)
                self.encoding_global_step = 0
                # dimensions
                self.state_dim = self.encoding_model.dim_z
                # observation buffer
                obs_dims = self.real_envs.observation_space.shape
                datatype = {'obs': {'dims': obs_dims}, 'next_obs': {'dims': obs_dims},
                            'actions': {'dims': [self.action_dim]}, 'rewards': {'dims': [1]}, 'masks': {'dims': [1]}}
                self.obs_buffer = Buffer(mbpo_cfg['real_buffer_size'], datatype)
                self.obs_buffer.to(self.device)
                # TODO: buffer entry dict check
            else:
                self.state_dim = np.prod(self.real_envs.observation_space.shape)

            # extract dim info
            datatype = {'states': {'dims': [self.state_dim]}, 'next_states': {'dims': [self.state_dim]},
                        'actions': {'dims': [self.action_dim]}, 'rewards': {'dims': [1]}, 'masks': {'dims': [1]}}

            # init normalizers
            self.state_normalizer = RunningNormalizer(self.state_dim)
            self.action_normalizer = RunningNormalizer(self.action_dim)
            self.state_normalizer.to(self.device)
            self.action_normalizer.to(self.device)

            # init env dynamics
            dynamics = EnsembleRDynamics(self.state_dim, self.action_dim, 1,
                                         mbpo_cfg['dynamics_hidden_dims'],
                                         mbpo_cfg['num_dynamics_networks'],
                                         mbpo_cfg['num_elite_dynamics_networks'],
                                         self.state_normalizer, self.action_normalizer)
            dynamics.to(self.device)

            # init env termination function
            term_fn = TerminationFn(
                self.state_dim, self.action_dim, mbpo_cfg['termination_fn_hidden_dims'])
            term_fn.to(self.device)

            # init sac agent (model-free algo)
            actor = Actor(self.state_dim, self.action_space, sac_cfg['actor_hidden_dims'],
                          state_normalizer=None, use_limited_entropy=False,
                          use_tanh_squash=True, use_state_dependent_std=True)
            actor.to(self.device)

            q_critic1 = QCritic(self.state_dim, self.action_space, sac_cfg['critic_hidden_dims'])
            q_critic2 = QCritic(self.state_dim, self.action_space, sac_cfg['critic_hidden_dims'])
            q_critic_target1 = QCritic(self.state_dim, self.action_space, sac_cfg['critic_hidden_dims'])
            q_critic_target2 = QCritic(self.state_dim, self.action_space, sac_cfg['critic_hidden_dims'])
            q_critic1.to(self.device)
            q_critic2.to(self.device)
            q_critic_target1.to(self.device)
            q_critic_target2.to(self.device)
            target_entropy = sac_cfg['target_entropy'] or \
                -np.prod(self.action_space.shape).item()
            self.agent = SAC(actor, q_critic1, q_critic2, q_critic_target1, q_critic_target2,
                             sac_cfg['batch_size'], sac_cfg['num_grad_steps'], env_cfg.gamma,
                             1.0, sac_cfg['actor_lr'], sac_cfg['critic_lr'], sac_cfg['soft_target_tau'],
                             target_entropy=target_entropy)

            # init fake env
            self.fake_env = make_fake_env(dynamics, term_fn)

            # init virtual env buffer
            self.base_virtual_buffer_sz = mbpo_cfg['rollout_batch_size'] * env_cfg.max_episode_steps * \
                mbpo_cfg['num_model_retain_epochs'] // mbpo_cfg['model_update_interval']
            self.virtual_buffer = Buffer(self.base_virtual_buffer_sz, datatype)
            self.virtual_buffer.to(self.device)
            self.agent.check_buffer(self.virtual_buffer)

            # init mbpo model
            self.model = HighDimMBPO(
                dynamics, term_fn, mbpo_cfg['dynamics_batch_size'],
                rollout_schedule=mbpo_cfg['rollout_schedule'], verbose=1,
                lr=mbpo_cfg['lr'], term_lr=mbpo_cfg['term_lr'],
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

            # init epochs (start - 1)
            self.epoch = -1

            # load checkpoints
            if ((not rl_cfg.model_load_path) ^ (not rl_cfg.buffer_load_path)):
                raise Exception("partial loading model or buffer may cause error")
            self._load()

            # init obs, reward and length
            self.real_obs = self.real_envs.reset()
            self.real_episode_rewards = deque(maxlen=30)
            self.real_episode_lengths = deque(maxlen=30)
            # warm up buffer
            self.warm_up()

            # log and summary
            logger = self.log_comp['logger']
            writer = self.log_comp['summary_writer'] if 'summary_writer' in self.log_comp else None

            logger.info('Env Observation Space: {}'.format(self.real_envs.observation_space))
            logger.info('Env Action Space: {}'.format(self.real_envs.action_space))
            logger.info('Env Observation Data Shape: {}'.format(self.real_obs.size()))
            logger.info('Env State Dim: {}'.format(self.state_dim))
            logger.info('Dynamics Model: \n{}'.format(dynamics))
            logger.info('Termination Function: \n{}'.format(term_fn))
            logger.info('Actor Model: \n{}'.format(actor))
            logger.info('Critic Model: \n')
            logger.info('1: \n{}'.format(q_critic1))
            logger.info('2: \n{}'.format(q_critic2))
            logger.info('target 1: \n{}'.format(q_critic_target1))
            logger.info('target 2: \n{}'.format(q_critic_target2))
            # fake_input = (torch.zeros(self.real_envs.observation_space.shape),
            #               torch.zeros(self.real_envs.action_space.shape))
            if self.encoding:
                logger.info('Encoding Model: \n{}'.format(self.encoding_model))
                writer.add_graph(
                    self.encoding_model, torch.zeros(
                        1, *self.encoding_model.input_size).to(self.device))
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

            start_epoch = self.epoch + 1
            for self.epoch in range(start_epoch, mbpo_cfg['num_total_epochs']):
                logger.info('Epoch {}:'.format(self.epoch))

                # update rollout length k
                self.model.update_rollout_length(self.epoch)
                # get encoder update interval
                if self.encoding:
                    encode_update_int = self.get_encode_update_interval(self.epoch)

                for i in range(env_cfg.max_episode_steps):
                    self.losses = {}

                    # record the normalizer missing updates
                    normalizer_update_samples = 0
                    if self.encoding and i % encode_update_int == 0:
                        # update encoding model
                        self.update_encoding_model()

                        # update state buffer by encoding obs buffer
                        self.encode_obs_buffer()

                        # update state and action normalizer
                        recent_states, recent_actions = itemgetter('states', 'actions') \
                            (self.real_buffer.get_recent_samples(self.real_buffer.size))
                        self.state_normalizer.update(recent_states)
                        self.action_normalizer.update(recent_actions)
                        normalizer_update_samples = 0

                    if i % mbpo_cfg['model_update_interval'] == 0:
                        # update predictive model
                        if normalizer_update_samples > 0:
                            recent_states, recent_actions = itemgetter('states', 'actions') \
                                (self.real_buffer.get_recent_samples(normalizer_update_samples))
                            self.state_normalizer.update(recent_states)
                            self.action_normalizer.update(recent_actions)
                            normalizer_update_samples = 0

                        self.losses.update(self.model.update(self.real_buffer))

                        # generate data in fake env
                        initial_states = next(self.real_buffer.get_batch_generator_inf(
                            mbpo_cfg['rollout_batch_size']))['states']
                        new_virtual_buffer_sz = self.base_virtual_buffer_sz * self.model.num_rollout_steps
                        self.virtual_buffer.resize(new_virtual_buffer_sz)
                        self.model.generate_data(self.fake_env, self.virtual_buffer,
                                                 initial_states, self.agent.actor)

                    # set start states
                    if self.encoding:
                        with torch.no_grad():
                            self.real_states = self.encode(self.real_obs)
                    else:
                        self.real_states = torch.flatten(self.real_obs).view(self.real_obs.size(0), -1)

                    # take actions in real env and add to buffer
                    with torch.no_grad():
                        real_actions = self.agent.actor.act(self.real_states)['actions']

                    real_next_obs, real_rewards, real_dones, real_infos = self.real_envs.step(
                        self.unwrap_action(real_actions))
                    real_masks = torch.tensor([[0.0] if done else [1.0] for done in real_dones], dtype=torch.float32)
                    if self.encoding:
                        # insert new observation data
                        self.obs_buffer.insert(obs=self.real_obs, actions=real_actions, rewards=real_rewards,
                                               masks=real_masks, next_obs=real_next_obs)
                        # encode obs into state and insert into buffer (re-encode current obs for consistency)
                        with torch.no_grad():
                            real_next_states = self.encode(real_next_obs)
                        self.real_buffer.insert(states=self.real_states, actions=real_actions, rewards=real_rewards,
                                                masks=real_masks, next_states=real_next_states)
                    else:
                        real_next_states = torch.flatten(real_next_obs).view(real_next_obs.size(0), -1)
                        self.real_buffer.insert(states=self.real_states, actions=real_actions, rewards=real_rewards,
                                                masks=real_masks, next_states=real_next_states)
                    normalizer_update_samples = normalizer_update_samples + 1

                    self.real_obs = real_next_obs
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
                        num_env_steps = self.epoch * env_cfg.max_episode_steps + i + mbpo_cfg['num_warmup_samples']
                        log_infos = {'/time_elapsed': time_elapsed, 'samples_collected': num_env_steps}

                        if len(self.real_episode_rewards) > 0:
                            log_infos['perf/ep_rew_real'] = np.mean(self.real_episode_rewards)
                            log_infos['perf/ep_len_real'] = np.mean(self.real_episode_lengths)
                        for k in self.losses.keys():
                            log_infos['loss/' + k] = self.losses[k]
                        log_and_write(log_infos, global_step=num_env_steps, logger=logger, writer=writer)

                if (self.epoch + 1) % rl_cfg.eval_interval == 0:
                    encoder = self.encoding_model if self.encoding else None
                    encode_fn = vae_encode if self.encoding else None
                    episode_rewards_real_eval, episode_lengths_real_eval = \
                        evaluate_atari(self.agent.actor, env_cfg.env_name, get_seed(), 10, self.eval_log_dir,
                                       self.device, encoder=encoder, encode_fn=encode_fn)
                    log_infos = {'perf/ep_rew_real_eval': np.mean(episode_rewards_real_eval),
                                 'perf/ep_len_real_eval': np.mean(episode_lengths_real_eval)}
                    log_and_write(log_infos, logger=logger, writer=writer,
                                  global_step=(self.epoch + 1) * env_cfg.max_episode_steps + mbpo_cfg['num_warmup_samples'])

                # break points
                if (self.epoch + 1) % rl_cfg.save_interval == 0:
                    extra = {'epoch': self.epoch}
                    if self.encoding:
                        extra['encoding_global_step'] = self.encoding_global_step
                    self._save(extra=extra)
        except Exception:
            # store buffers and losses when error occurred
            self.real_buffer.save(os.path.join(self.monitor_dir, 'real_state_buffer.pt'))
            self.virtual_buffer.save(os.path.join(self.monitor_dir, 'virtual_state_buffer.pt'))
            if self.encoding:
                self.obs_buffer.save(os.path.join(self.monitor_dir, 'real_obs_buffer.pt'))
            torch.save(self.losses, os.path.join(self.monitor_dir, 'losses.pt'))

            raise

    def after_run(self, **kwargs):
        """cleaning up needed be done after run the experiment"""
        # save model, plot losses
        try:
            self.real_envs.close()
            if "summary_writer" in self.log_comp:
                writer = self.log_comp["summary_writer"]
                writer.close()
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
                       'q_critic_target2': self.agent.target_q_critic2.state_dict(),
                       'term_fn': self.model.term_fn.state_dict()}
        # store the cfgs into the state dict pack for convenience
        cfgs = {}
        for cfg_key in save_model_cfg.store_cfgs:
            cfgs[cfg_key] = self.exp_configs[cfg_key].raw
        if cfgs:
            state_dicts['exp_configs'] = cfgs

        torch.save(state_dicts, self.save_dir + '/state_dicts.pt')
        if self.encoding:
            self.obs_buffer.save(self.save_dir + '/real_buffer.pt')
        else:
            self.real_buffer.save(self.save_dir + '/real_buffer.pt')

        # store some extra info (e.g. epoch)
        if extra is not None:
            torch.save(extra, self.save_dir + '/extra.pt')

        if self.encoding:
            # save the encoding model (just model info and model config)
            encoding_cfg = rl_cfg.encoding_config
            model_cfg = encoding_cfg.model_config

            extra = get_extra_setting(model_cfg.model_name, self.encoding_model)

            torch.save({
                "exp_configs": {"model": model_cfg.raw},
                "model_state_dict": self.encoding_model.state_dict(),
                "extra": extra}, self.encoding_model_file)
