# experiment related util functions
import matplotlib
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