from torch import optim


def configure_optimizer_vae(model, optimizer_config):
    """initialize optimizer for basic vae
    :param model: the vae model object
    :param optimizer_config: defined in train_config
    :return: optimizer object
    """
    optimizer = optim.Adam(model.parameters(), lr=optimizer_config.lr)

    return optimizer


def configure_optimizer_factor_vae(model, optimizer_config):
    """initialize optimizers for factor vae
    :param model: the factor vae model object
    :param optimizer_config: defined in train_config
    :return: a dict, mapping optimizer name to corresponding optimizer object
    """
    optimizers = {}

    for name in optimizer_config.subs.keys():
        sub_optim_cfg = optimizer_config.subs[name]
        optimizers[name] = optim.Adam(getattr(model, name).parameters(), lr=sub_optim_cfg.lr)

    return optimizers


configure_optimizer_handlers = {
    "VAE": configure_optimizer_vae,
    "ConvVAE": configure_optimizer_vae,
    "BetaVAE": configure_optimizer_vae,
    "ConvBetaVAE": configure_optimizer_vae,
    "BetaTCVAE": configure_optimizer_vae,
    "ConvBetaTCVAE": configure_optimizer_vae,
    "FactorVAE": configure_optimizer_factor_vae,
    "ConvFactorVAE": configure_optimizer_factor_vae,
    "SparseVAE": configure_optimizer_vae,
    "ConvSparseVAE": configure_optimizer_vae,
    "JointVAE": configure_optimizer_vae,
    "ConvJointVAE": configure_optimizer_vae
}


def adapt_configure_optimizer(model_name, model, optimizer_config):
    """configure optimizer for different model"""
    if model_name in configure_optimizer_handlers:
        return configure_optimizer_handlers[model_name](model, optimizer_config)
    else:
        raise Exception("no configure optimizer handler for model {}".format(model_name))
