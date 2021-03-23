import torch

from config import *
from models.vaes import *


def load_extra_setting_vae(model, extra_args):
    """load the saved extra args for standard VAE object (and some other VAEs)"""
    pass


def load_extra_setting_sparse_vae(model, extra_args):
    """load the saved extra args for sparse VAE object"""
    model.c = extra_args["model_c"]


def load_extra_setting_joint_vae(model, extra_args):
    """load the saved extra args for joint VAE object"""
    if "cont_cap" in extra_args:
        model.cont_cap_current = extra_args["cont_cap"]
    if "disc_cap" in extra_args:
        model.disc_cap_current = extra_args["disc_cap"]


def load_extra_setting_error(model, extra_args):
    """error handler for loading extra setting"""
    raise NotImplementedError


load_extra_setting_handlers = {
    "VAE": load_extra_setting_vae,
    "ConVAE": load_extra_setting_vae,
    "BetaVAE": load_extra_setting_vae,
    "ConvBetaVAE": load_extra_setting_vae,
    "BetaTCVAE": load_extra_setting_vae,
    "ConvBetaTCVAE": load_extra_setting_vae,
    "FactorVAE": load_extra_setting_vae,
    "ConvFactorVAE": load_extra_setting_vae,
    "SparseVAE": load_extra_setting_sparse_vae,
    "ConvSparseVAE": load_extra_setting_sparse_vae,
    "JointVAE": load_extra_setting_joint_vae,
    "ConvJointVAE": load_extra_setting_joint_vae
}


def load_model(filepath):
    """load model file
    :param filepath: the file path of the model
    :return: a dict containing model object, config objects and others
    """
    model_checkpoint = torch.load(filepath)
    exp_configs_dict = model_checkpoint["exp_configs"]
    model_state_dict = model_checkpoint["model_state_dict"]
    extra = model_checkpoint["extra"]

    # restore the config objects
    exp_configs = {}
    if "general" in exp_configs_dict:
        exp_configs["general"] = GeneralConfiguration(**exp_configs_dict["general"])
    if "dataset" in exp_configs_dict:
        exp_configs["dataset"] = DatasetConfiguration(**exp_configs_dict["dataset"])
    if "model" in exp_configs_dict:
        exp_configs["model"] = ModelConfiguration(**exp_configs_dict["model"])
    if "train" in exp_configs_dict:
        exp_configs["train"] = TrainConfiguration(**exp_configs_dict["train"])

    # restore the model
    model = eval(exp_configs["model"].model_name)(**exp_configs["model"].model_args)
    model.load_state_dict(model_state_dict)
    if exp_configs["model"].model_name in load_extra_setting_handlers:
        load_extra_setting_handlers[exp_configs["model"].model_name](model, extra)
    else:
        raise Exception("no extra setting handler for the model {}".format(exp_configs["model"].model_name))

    return {
        "model": model,
        "configs": exp_configs
    }
