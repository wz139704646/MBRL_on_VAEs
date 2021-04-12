# experiment related util functions
import torch
import random
import logging
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_losses(losses, save_path=None, layout='v'):
    """plot the losses change over epoches
    :param losses: a dict mapping loss name to list of loss values (up to 9 categories)
    :param save_dir: if not None, save the fig at the path
    :param layout: currently 'v' and 'h' supported, specify how the subplot placed
    """
    cates = list(losses.keys())
    num = len(cates)
    assert num <= 9

    # default vertical layout
    nrow = 1
    ncol = 1
    if num >= 3:
        nrow = 3
        ncol = (num-1) // 3 + 1
    else:
        nrow = num
        ncol = 1

    if layout == 'v':
        # vertical layout
        pass
    elif layout == 'h':
        # horizontal layout
        nrow, ncol = ncol, nrow
    else:
        raise NotImplementedError

    fig = plt.figure(figsize=(12, 12))
    for i in range(num):
        idx = i + 1
        if layout == 'v':
            col = i // nrow # the subplot col no.
            row = i % nrow # the subplot row no.
            idx = row * ncol + col + 1
        elif layout == 'h':
            pass

        ax = fig.add_subplot(nrow, ncol, idx)
        ax.set_title(cates[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.plot(losses[cates[i]])

    if save_path is not None:
        plt.savefig(save_path)


def add_salt_pepper_noise(img, density, scale=True):
    """add salt and pepper noise according to the density
    :param img: the image to add noise (shape [C x H x W])
    :return: ndarray with salt and pepper noise added
    """
    x = np.array(img)
    c, h, w = x.shape
    Nd = density
    Sd = 1 - Nd

    # add pepper to 0, add salt to 1
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[Nd/2.0, Nd/2.0, Sd])
    mask = np.repeat(mask, c, axis=0)
    x[mask == 0] = 0. if scale else 0
    x[mask == 1] = 1. if scale else 255

    return x


def add_gaussian_noise(img, mean, var, amplitude, scale=True):
    """add gaussian noise
    :param img: the image to add noise (shape [C x H x W])
    :param mean: param of the gaussian distribution
    :param var: param of the gaussian distribution
    :param amplitude: the amplitude of the noise
    """
    x = np.array(img)
    c, h, w = x.shape

    # generate gaussian noise
    N = amplitude * np.random.normal(loc=mean, scale=var, size=(1, h, w))
    N = np.repeat(N, c, axis=0)

    # add noise to image
    x = x + N
    if scale:
        x[x > 1.] = 1.
        x[x < 0.] = 0.
    else:
        x[x > 255] = 255
        x[x < 0] = 0

    return x


def set_seed(seed: int, strict=False):
    """set random seed
    :param seed: int, random seed
    :param strict: bool, whether make it deterministic strictly
    """
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(2 ** 30))
    random.seed(np.random.randint(2 ** 30))
    try:
        torch.cuda.manual_seed_all(np.random.randint(2 ** 30))
        if strict:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass


def get_seed():
    return random.randint(0, 2 ** 32 - 1)


def log_and_write(log_infos, global_step, logger, writer=None):
    """log kv log infos and writer them into tensorboard
    :param log_infos: key value log infos
    :param global_step: int, for step record in summary writer
    :param logger: the logger to log the infos
    :param writer: the summary writer used (not used if None).
        writer only records the infos with keys containing '/'
    """
    logger = logger or logging.getLogger()

    for k in log_infos.keys():
        logger.info('{}: {}'.format(k.split('/')[-1], log_infos[k]))
        if writer and k.find('/') > -1:
            writer.add_scalar(k, log_infos[k], global_step=global_step)
