import os
import torch
import random
import numpy as np
from scipy import stats as st
from torchvision.utils import save_image

from envs import *
from models.vaes.beta_vae import ConvBetaVAE


import sys
sys.path.append(".")
sys.path.append("..")
from utils.model import latent_traversal_detailed


def set_basic_config():
    """set basic test configuration"""
    conf = {}
    conf['checkpoint_path'] = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/GridWorld_ch3/t1614753837-betaVAE_epoch60_z20_beta3.pth.tar')

    return conf


def set_test_config():
    """set test configuration"""
    conf = set_basic_config()

    conf['save_dir'] = os.path.join(
        os.path.dirname(__file__),
        'results/latent_traversal/GridWorld_betaVAE_epoch60_z15_beta3')
    # conf['traversal_vals'] = torch.arange(-3, 3.1, 6./15).tolist()

    # the values to traverse (according to normal distribution)
    conf['traversal_vals'] = st.norm.ppf(np.linspace(0.03, 0.997, 20))

    if not os.path.exists(conf['save_dir']):
        os.makedirs(conf['save_dir'])

    return conf


if __name__ == '__main__':
    test_conf = set_test_config()
    checkpoint_path = test_conf['checkpoint_path']
    save_dir = test_conf['save_dir']
    traversal_vals = torch.tensor(test_conf['traversal_vals'])

    model_checkpoint = torch.load(checkpoint_path)
    train_args = model_checkpoint['train_args']
    train_conf = model_checkpoint['train_config']
    model_state_dict = model_checkpoint['state_dict']

    img_size = train_conf['env_config'][train_args.env_name]['image_size']
    model = ConvBetaVAE(img_size, train_args.channels, train_args.hidden_size,
                        train_args.dim_z, train_args.beta)
    model.load_state_dict(model_state_dict)

    # traverse all using random codes
    latent_traversal_detailed(model, traversal_vals, img_size, save_dir,
                              tag="{}_traversal_random".format(train_args.env_name), num=10)

    # traverse all using codes encoded from dataset
    env_init_args = train_conf["env_init_args"][train_args.env_name]
    env = eval(train_args.env_name)(**env_init_args)
    all_samples = torch.tensor(env.traverse())
    base_input = all_samples[random.sample(range(0, all_samples.size(0)), 10)]
    latent_traversal_detailed(model, traversal_vals, img_size, save_dir,
                              tag="{}_traversal_sample".format(train_args.env_name),
                              base_input=base_input)