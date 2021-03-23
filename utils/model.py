# model related functinos
import os
import time
import torch
from torchvision.utils import save_image


def save_(model, save_dir, args=None, config=None, extra=None, comment=None):
    """save the model accompany with its args and configuration
    :param model: the vae model object
    :param save_dir: directory to put the saved files
    :param args: the input arguments when training
    :param config: the config of the training
    :param extra: others to save
    :param comment: tag at the end of filename
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "t{}".format(int(time.time()))
    if comment is not None:
        filename = "{}-{}".format(filename, comment)
    filename = os.path.join(save_dir, filename + '.pth.tar')

    torch.save({'train_args': args,
                'train_config': config,
                'state_dict': model.state_dict(),
                'extra': extra}, filename)


def latent_traversal_detailed(model, traversal_vals, img_size, save_dir,
                              tag='latent_traversal', base_input=None, num=10):
    """traverse the latent space for each dimension
    :param model: the vae model
    :param traversal_vals: the traversal values for each dimension
    :param tag: the tag string in the traversal results filenames
    :param base_input: the base original input for the model to encode,
                       default None (will sample num samples from latent
                       space under normal distribution)
    :param num: number of base samples (only work when base_input is None)
    """
    model.eval()

    with torch.no_grad():
        dim_z = model.dim_z
        if base_input is not None:
            codes = model.encode(base_input)[0]
        else:
            codes = torch.randn(num, dim_z)

        # store the original image
        dec = model.decode(codes)
        save_image(dec, os.path.join(save_dir, "{}_original.png".format(tag)), nrow=num, pad_value=1)

        num_vals = traversal_vals.size(0)
        for i in range(codes.size(0)):
            # traversal for each code
            c = codes[i].clone()
            imgs = []
            for j in range(dim_z):
                # traverse each dimension
                cmat = torch.cat([c.clone().unsqueeze(0) for _ in range(num_vals)])
                cmat[:, j] = traversal_vals
                dec = model.decode(cmat)
                dec = dec.view(-1, *img_size)
                print("code {}, dim {}, codes: \n{}\nget image serize: \n{}".format(i, j, cmat, dec))
                imgs.append(dec)

            save_image(torch.cat(imgs).cpu(),
                       os.path.join(save_dir, "{}_{}.png".format(tag, i)),
                       nrow=num_vals, pad_value=1)
