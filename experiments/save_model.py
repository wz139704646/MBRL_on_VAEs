def get_extra_setting_vae(model):
    """return the extra args to save for standard VAE object (and some other VAEs)"""
    return None


def get_extra_setting_sparse_vae(model):
    """return the saved extra args for sparse VAE object"""
    return {
        "model_c": model.c
    }


def get_extra_setting_joint_vae(model):
    """load the saved extra args for joint VAE object"""
    extra = {}
    if model.dim_cont > 0:
        extra["cont_cap"] = model.cont_cap_current
    if model.dim_disc > 0:
        extra["disc_cap"] = model.disc_cap_current

    return extra


get_extra_setting_handlers = {
    "VAE": get_extra_setting_vae,
    "ConVAE": get_extra_setting_vae,
    "BetaVAE": get_extra_setting_vae,
    "ConvBetaVAE": get_extra_setting_vae,
    "BetaTCVAE": get_extra_setting_vae,
    "ConvBetaTCVAE": get_extra_setting_vae,
    "FactorVAE": get_extra_setting_vae,
    "ConvFactorVAE": get_extra_setting_vae,
    "SparseVAE": get_extra_setting_sparse_vae,
    "ConvSparseVAE": get_extra_setting_sparse_vae,
    "JointVAE": get_extra_setting_joint_vae,
    "ConvJointVAE": get_extra_setting_joint_vae
}


def get_extra_setting(model_name, model):
    """return the extra setting for saving model"""
    if model_name in get_extra_setting_handlers:
        return get_extra_setting_handlers[model_name](model)
    else:
        raise Exception("no getting extra setting handler for the model {}".format(model_name))