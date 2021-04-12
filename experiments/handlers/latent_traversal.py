import os
import torch
import numpy as np
from scipy import stats as st
from torchvision.utils import save_image


def latent_traversal_vae(model, base_codes, save_cfg, **kwargs):
    """latent traversal function for standard vae (and some other models)
    :param model: the vae model object
    :param base_codes: the codes traversal based on
    :param save_cfg: configuration for saving results
    :param kwargs: some unique params needed for traversal.
        indices: the indices of dimensions to traverse
        traversal_range: tuple, the range to traverse, (prob_low, prob_high, size)
    """
    save_dir = save_cfg.default_dir
    tag = save_cfg.tag
    indices = kwargs["indices"]
    traversal_range = kwargs["traversal_range"]
    traversal_vals = st.norm.ppf(np.linspace(*traversal_range))

    # save the base codes image
    sample_out = model.decoded_to_output(model.decode(base_codes.clone().detach()))
    save_image(sample_out.cpu(),
               os.path.join(save_dir, "base_samples.png"),
               nrow=1, pad_value=1)

    for idx in indices:
        # generate one image for each index
        imgs = []
        for i in range(base_codes.size(0)):
            c = base_codes[i].clone().detach()
            c_mat = torch.cat(
                [c.clone().detach().unsqueeze(0) for _ in range(len(traversal_vals))])
            # set different vals
            c_mat[:, idx] = torch.tensor(traversal_vals)
            out = model.decoded_to_output(model.decode(c_mat))

            imgs.append(out)

        save_image(torch.cat(imgs).cpu(),
                   os.path.join(save_dir, "{}_idx{}.png".format(tag, idx)),
                   nrow=len(traversal_vals), pad_value=1)


def latent_traversal_joint_vae(model, base_codes, save_cfg, **kwargs):
    """latent traversal function for joint vae
    :param model: the vae model object
    :param base_codes: the codes traversal based on
    :param save_cfg: configuration for saving results
    :param kwargs: some unique params needed for traversal.
        cont_indices: the indices of continuous dimensions to traverse
        cont_traversal_range: tuple, the range to traverse, (prob_low, prob_high, size)
        disc_indices: the indices of discrete variables to traverse (variable index)
    """
    cont_indices = kwargs["cont_indices"] if "cont_indices" in kwargs else []
    cont_traversal_range = kwargs["cont_traversal_range"] \
        if "cont_traversal_range" in kwargs else None
    disc_indices = kwargs["disc_indices"] if "disc_indices" in kwargs else []

    # save the base codes image
    save_dir = save_cfg.default_dir
    sample_out = model.decoded_to_output(model.decode(base_codes.clone().detach()))
    save_image(sample_out.cpu(),
               os.path.join(save_dir, "base_samples.png"),
               nrow=1, pad_value=1)

    # traverse continuous variables
    for cont_idx in cont_indices:
        traverse_joint_vae_cont(model, base_codes, cont_idx, cont_traversal_range, save_cfg)

    # traverse discrete variables
    for disc_idx in disc_indices:
        traverse_joint_vae_disc(model, base_codes, disc_idx, save_cfg)


def traverse_joint_vae_cont(model, base_codes, cont_idx, traversal_range, save_cfg):
    """traverse continuous variable of joint vae"""
    save_dir = save_cfg.default_dir
    tag = save_cfg.tag
    traversal_vals = st.norm.ppf(np.linspace(*traversal_range))

    imgs_cont = []
    for i in range(base_codes.size(0)):
        c = base_codes[i].clone().detach()
        c_mat = torch.cat(
            [c.clone().detach().unsqueeze(0) for _ in range(len(traversal_vals))])
        # set different values
        c_mat[:, cont_idx] = torch.tensor(traversal_vals)
        out = model.decoded_to_output(model.decode(c_mat))

        imgs_cont.append(out)

    save_image(torch.cat(imgs_cont).cpu(),
               os.path.join(save_dir, "{}_contIdx{}.png".format(tag, cont_idx)),
               nrow=len(traversal_vals), pad_value=1)


def traverse_joint_vae_disc(model, base_codes, disc_idx, save_cfg):
    """traverse discrete variable of joint vae"""
    save_dir = save_cfg.default_dir
    tag = save_cfg.tag
    dim_disc = model.latent_disc[disc_idx]
    pre_dims = model.dim_cont + sum(model.latent_disc[:disc_idx])
    # intialize codes, set discrete variable to all 0
    codes = base_codes.clone().detach()
    codes[:, pre_dims:(pre_dims + dim_disc)] = 0.

    imgs_disc = []
    for i in range(codes.size(0)):
        c = codes[i].clone().detach()
        c_mat = torch.cat([c.clone().detach().unsqueeze(0) for _ in range(dim_disc)])
        # set different values
        c_mat[np.arange(dim_disc), pre_dims + np.arange(dim_disc)] = 1.
        out = model.decoded_to_output(model.decode(c_mat))

        imgs_disc.append(out)

    save_image(torch.cat(imgs_disc).cpu(),
               os.path.join(save_dir, "{}_discIdx{}.png".format(tag, disc_idx)),
               nrow=dim_disc, pad_value=1)


def latent_traversal_sparse_vae(model, base_codes, save_cfg, **kwargs):
    """latent traversal function for sparse vae
    :param model: the sparse vae model object
    :param base_codes: the codes traversal based on
    :param save_cfg: configuration for saving results
    :param kwargs: some unique params needed for traversal.
        indices: the indices of dimensions to traverse
            (choose the highest value dimension if None)
        traversal_range: tuple, the range to traverse, (prob_low, prob_high, size)
    """
    save_dir = save_cfg.default_dir
    tag = save_cfg.tag
    indices = kwargs["indices"]
    traversal_range = kwargs["traversal_range"]
    traversal_vals = st.norm.ppf(np.linspace(*traversal_range))

    # save the base codes image
    sample_out = model.decoded_to_output(model.decode(base_codes.clone().detach()))
    save_image(sample_out.cpu(),
               os.path.join(save_dir, "base_samples.png"),
               nrow=1, pad_value=1)

    if indices is None:
        # select the highest value dimension to traverse
        indices = []
        for i in range(base_codes.size(0)):
            z = base_codes[i]
            max_ind = torch.argmax(z).item()
            indices.append(max_ind)

    for idx in indices:
        # generate one image for each index
        imgs = []
        for i in range(base_codes.size(0)):
            c = base_codes[i].clone().detach()
            c_mat = torch.cat(
                [c.clone().detach().unsqueeze(0) for _ in range(len(traversal_vals))])
            # set different vals
            c_mat[:, idx] = torch.tensor(traversal_vals)
            out = model.decoded_to_output(model.decode(c_mat))

            imgs.append(out)

        save_image(torch.cat(imgs).cpu(),
                   os.path.join(save_dir, "{}_idx{}.png".format(tag, idx)),
                   nrow=len(traversal_vals), pad_value=1)


def latent_traversal_error(model, base_codes, save_cfg, **kwargs):
    raise NotImplementedError


latent_traversal_handlers = {
    "VAE": latent_traversal_vae,
    "ConVAE": latent_traversal_vae,
    "BetaVAE": latent_traversal_vae,
    "ConvBetaVAE": latent_traversal_vae,
    "BetaTCVAE": latent_traversal_vae,
    "ConvBetaTCVAE": latent_traversal_vae,
    "FactorVAE": latent_traversal_vae,
    "ConvFactorVAE": latent_traversal_vae,
    "SparseVAE": latent_traversal_sparse_vae,
    "ConvSparseVAE": latent_traversal_sparse_vae,
    "JointVAE": latent_traversal_joint_vae,
    "ConvJointVAE": latent_traversal_joint_vae
}


def adapt_latent_traversal(model_name, model, base_codes, save_cfg, **kwargs):
    if model_name in latent_traversal_handlers:
        latent_traversal_handlers[model_name](model, base_codes, save_cfg, **kwargs)
    else:
        raise Exception("no latent traversal handler for the model {}".format(model_name))