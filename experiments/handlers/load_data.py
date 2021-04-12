import os
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


load_data_handlers = {
    "VAE": load_data_vae,
    "ConVAE": load_data_vae,
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


def adapt_load_data(model_name, dataset_file, dataset_cfg, **kwargs):
    """adapt loading data function according to model name"""
    if model_name in load_data_handlers:
        return load_data_handlers[model_name](dataset_file, dataset_cfg, **kwargs)
    else:
        raise Exception("no load data handler for the model {}".format(model_name))
