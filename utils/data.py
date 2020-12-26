import os
import math
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class CustomTensorDataset(Dataset):
    """Custom class for reading tensor datasets"""

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def prepare_data_dsprites(batch_size, data_dir, shuffle=True, **kwargs):
    """load dsprites data (since the data are download via url, no train or test option)
    :param data_dir: the root directory path
    :param kwargs: other arguments for the return dataloader
    """
    data_file = "dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    data_file = os.path.join(data_dir, data_file)

    # load dataset
    data = np.load(data_file, encoding='bytes')
    data = torch.from_numpy(data['imgs']).unsqueeze(
        1).float()  # use image data
    dset = CustomTensorDataset(data)

    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def prepare_data_mnist(batch_size, data_dir, train=True, shuffle=True, **kwargs):
    """load MNIST data via torchvision
    :param data_dir: the root directory path
    :param train: whether the data is used for training (false means testing)
    :param kwargs: other arguments for the return dataloader
    """

    return DataLoader(datasets.MNIST(data_dir, train=train, download=True,
                                     transform=transforms.ToTensor()),
                      batch_size=batch_size, shuffle=shuffle, **kwargs)


def prepare_data_custom_tensor(batch_size, data_file=None, data=None, shuffle=True, div=None, **kwargs):
    """load custom tensor dataset in memory or disk
    :param data_file: read data from this file or save the data into this file
    :param data: data in memory (tensor form)
    :param div: a list of float, sumed up to 1, used to divide the dataset
    :param kwargs: other arguments for the return dataloader
    :return: a list of dataloader
    """
    if any([data is not None, data_file is not None]):
        # valid arguments
        if data is not None:
            # use data in memory
            if data_file is not None:
                # save the data file
                save_dir = os.path.dirname(data_file)
                if not os.path.exists(save_dir):
                    # ensure directory exists
                    os.makedirs(save_dir)
                torch.save(data, data_file)
        else:
            # use data in disk
            data = torch.load(data_file)
        dset = CustomTensorDataset(data)
        dataloaders = []

        if div is not None and sum(div) == 1:
            # divide the dataset
            total_len = len(dset)
            start = 0
            for per in div:
                tmp_len = math.floor(total_len * per)
                dataloaders.append(DataLoader(
                    dset[start:start+tmp_len], batch_size=batch_size, shuffle=shuffle, **kwargs))
                start += tmp_len
        else:
            # return the list containing a signle dataloader
            dataloaders.append(DataLoader(
                dset, batch_size=batch_size, shuffle=shuffle, **kwargs))

        return dataloaders
