import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from env_collect import collect_data
from utils.data import prepare_data_custom_tensor


def load_data_vae(dataset_file, dataset_cfg, **kwargs):
    """load data for vae training and testing
    :param dataset_file: the dataset file path, load from or store to
    :param dataset_cfg: the dataset configuration in exp configurations
    :param kwargs: kwargs for dataloader generation
    :return: tuple of dataloaders
    """
    if os.path.exists(dataset_file):
        # load from disk
        return tuple(prepare_data_custom_tensor(dataset_cfg.load_config.batch_size,
                                                data_file=dataset_file,
                                                shuffle=dataset_cfg.load_config.shuffle,
                                                div=dataset_cfg.load_config.division, **kwargs))
    elif dataset_cfg.type == "collect":
        # collect from env
        data = collect_data(dataset_cfg.collect_config)
        return tuple(prepare_data_custom_tensor(dataset_cfg.load_config.batch_size,
                                                data_file=dataset_file, data=data,
                                                shuffle=dataset_cfg.load_config.shuffle,
                                                div=dataset_cfg.load_config.division, **kwargs))
    else:
        raise Exception("unable to prepare data")


def load_data_factor_vae(dataset_file, dataset_cfg, **kwargs):
    """load data for vae training and testing
    :param dataset_file: the dataset file path, load from or store to
    :param dataset_cfg: the dataset configuration in exp configurations
    :param kwargs: kwargs for dataloader generation
    :return: tuple of list of dataloaders
    """
    if os.path.exists(dataset_file):
        # load from disk
        dataloaders = prepare_data_custom_tensor(dataset_cfg.load_config.batch_size,
                                                data_file=dataset_file,
                                                shuffle=dataset_cfg.load_config.shuffle,
                                                div=dataset_cfg.load_config.division, **kwargs)
    elif dataset_cfg.type == "collect":
        # collect from env
        data = collect_data(dataset_cfg.collect_config)
        dataloaders = prepare_data_custom_tensor(dataset_cfg.load_config.batch_size,
                                                data_file=dataset_file, data=data,
                                                shuffle=dataset_cfg.load_config.shuffle,
                                                div=dataset_cfg.load_config.division, **kwargs)
    else:
        raise Exception("unable to prepare data")

    # add a shuffled dataloader copy accompany with the original data loader
    return tuple([
        [
            loader,
            DataLoader(loader.dataset, batch_size=dataset_cfg.load_config.batch_size,
                       shuffle=True, **kwargs)
        ] for loader in dataloaders
    ])


def get_data_gen_with_buffer_vae(buffer, batch_size, indices, gen_type='epoch'):
    """load data from buffer for vae training/testing
    :param buffer: buffer that stores the data
    :param batch_size: a single data is a batch of datapoints with size of batch_size
    :param indices: the indices of datas for each part
    :param gen_type: 'epoch' or 'inf' generator
    :return: a tuple of generator
    """
    gens = []

    for ind in indices:
        if gen_type == 'epoch':
            gen = buffer.get_batch_generator_epoch(batch_size, ind)
        else:
            gen = buffer.get_batch_generator_inf(batch_size, ind)
        gens.append(gen)

    return tuple(gens)


def get_data_gen_with_buffer_factor_vae(buffer, batch_size, indices, gen_type='epoch'):
    """load data from buffer for factor vae training/testing
    :param buffer: buffer that stores the data
    :param batch_size: a single data is a batch of datapoints with size of batch_size
    :param indices: the indices of datas for each part
    :param gen_type: 'epoch' or 'inf' generator
    :return: a tuple of generators
    """
    gens = []

    for ind in indices:
        if gen_type == 'epoch':
            gen1 = buffer.get_batch_generator_epoch(batch_size, ind)
            gen2 = buffer.get_batch_generator_epoch(batch_size, ind)
        else:
            gen1 = buffer.get_batch_generator_inf(batch_size, ind)
            gen2 = buffer.get_batch_generator_inf(batch_size, ind)
        gens.append([gen1, gen2]) # factor need 2 generator in one process

    return tuple(gens)


load_data_handlers = {
    "VAE": load_data_vae,
    "ConvVAE": load_data_vae,
    "BetaVAE": load_data_vae,
    "ConvBetaVAE": load_data_vae,
    "BetaTCVAE": load_data_vae,
    "ConvBetaTCVAE": load_data_vae,
    "FactorVAE": load_data_factor_vae,
    "ConvFactorVAE": load_data_factor_vae,
    "SparseVAE": load_data_vae,
    "ConvSparseVAE": load_data_vae,
    "JointVAE": load_data_vae,
    "ConvJointVAE": load_data_vae
}
get_data_gen_with_buffer_handlers = {
    "VAE": get_data_gen_with_buffer_vae,
    "ConvVAE": get_data_gen_with_buffer_vae,
    "BetaVAE": get_data_gen_with_buffer_vae,
    "ConvBetaVAE": get_data_gen_with_buffer_vae,
    "BetaTCVAE": get_data_gen_with_buffer_vae,
    "ConvBetaTCVAE": get_data_gen_with_buffer_vae,
    "FactorVAE": get_data_gen_with_buffer_factor_vae,
    "ConvFactorVAE": get_data_gen_with_buffer_factor_vae,
    "SparseVAE": get_data_gen_with_buffer_vae,
    "ConvSparseVAE": get_data_gen_with_buffer_vae,
    "JointVAE": get_data_gen_with_buffer_vae,
    "ConvJointVAE": get_data_gen_with_buffer_vae
}


def adapt_load_data(model_name, dataset_file, dataset_cfg, **kwargs):
    """adapt loading data function according to model name"""
    if model_name in load_data_handlers:
        return load_data_handlers[model_name](dataset_file, dataset_cfg, **kwargs)
    else:
        raise Exception("no load data handler for the model {}".format(model_name))


def adapt_get_data_gen_with_buffer(model_name, buffer, batch_size, indices, gen_type='epoch'):
    """adapt getting data generator function according to model name"""
    if model_name in get_data_gen_with_buffer_handlers:
        return get_data_gen_with_buffer_handlers[model_name](buffer, batch_size, indices, gen_type)
    else:
        raise Exception("no get data generator with buffer handler for the model {}".format(model_name))