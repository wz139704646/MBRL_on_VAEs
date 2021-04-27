from __future__ import annotations

from typing import Dict, Generator, Optional
from operator import itemgetter
import logging

import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data.sampler import RandomSampler, BatchSampler, SubsetRandomSampler


# noinspection DuplicatedCode
class SimpleUniversalBuffer:
    def __init__(self, buffer_size: int, entry_dict: Dict[str, dict], **kwargs):
        self.entry_infos = entry_dict.copy()
        self.entries = self.entry_infos.keys()
        self.entry_num_classes = {}
        self.device = torch.device('cpu')
        self.buffer_size = buffer_size
        self.size = 0
        self.index = 0

        for name, properties in entry_dict.items():
            dims = properties.get('dims')
            dtype = torch.float32
            if isinstance(dims, int):
                self.entry_num_classes[name] = dims
                dims = [1]
                dtype = torch.int32
            self.__dict__[name] = torch.zeros(buffer_size, *dims, dtype=dtype)

        for name, value in kwargs.items():
            setattr(self, name, value)

    def to(self, device, inplace=False):
        self.device = device
        if inplace:
            for entry in self.entry_infos.keys():
                self.__dict__[entry] = self.__dict__[entry].to(device)

    def insert(self, **kwargs):
        lengths = []
        for (name, value) in kwargs.items():
            lengths.append(value.shape[0])
            self._insert_sequence(name, value, self.index)
        assert len(set(lengths)) == 1
        self.size = min(self.size + lengths[0], self.buffer_size)
        self.index = (self.index + lengths[0]) % self.buffer_size

    def clear(self):
        self.index = self.size = 0

    def _insert_sequence(self, name, values, index):
        length = values.shape[0]
        if index + length <= self.buffer_size:
            self.__dict__[name][index: index + length].copy_(values)
        else:
            self._insert_sequence(name, values[:self.buffer_size - index], index)
            self._insert_sequence(name, values[self.buffer_size - index:], 0)

    def add_buffer(self, buffer: SimpleUniversalBuffer):
        self._add_offpolicy_buffer(buffer)

    def save(self, save_path):
        save_result = {'index': self.index, 'size': self.size, 'entry_infos': self.entry_infos,
                       'entry_num_classes': self.entry_num_classes}
        for name in set(self.entry_infos.keys()):
            save_result[name] = self.__dict__[name].cpu().clone()
        torch.save(save_result, save_path)

    def load(self, load_path):
        buffer_infos = torch.load(load_path)
        self.index, self.size, entry_infos, self.entry_num_classes = \
            itemgetter('index', 'size', 'entry_infos', 'entry_num_classes')(buffer_infos)
        assert entry_infos == self.entry_infos, logging.error('loaded entries: {}\nbuffer entries: {}'.
                                                             format(entry_infos, self.entry_infos))
        sizes = []
        for name in self.entry_infos:
            self.__dict__[name] = buffer_infos[name]
            sizes.append(buffer_infos[name].shape[0])
        assert len(set(sizes)) == 1
        self.resize(self.buffer_size)

    def get_batch_generator_inf(self, batch_size: Optional[int], ranges=None) -> Generator:
        ranges = range(self.size) if ranges is None else ranges
        batch_size = batch_size or len(ranges)
        while True:
            indices = np.fromiter(RandomSampler(ranges, replacement=True, num_samples=batch_size), np.int32)
            batch_data = {}
            for name in self.entry_infos.keys():
                batch_data[name] = self.__dict__[name][indices].clone().\
                    to(self.device)
                if name in self.entry_num_classes and getattr(self, 'use_onehot_output', False):
                    batch_data[name] = one_hot(batch_data[name].long(), num_classes=self.entry_num_classes[name]).\
                        squeeze(-2).float()
            yield batch_data

    def get_batch_generator_epoch(self, batch_size: Optional[int], ranges=None) -> Generator:
        ranges = range(self.size) if ranges is None else ranges
        batch_size = batch_size or len(ranges)
        sampler = BatchSampler(SubsetRandomSampler(ranges), batch_size, drop_last=True)
        for indices in sampler:
            batch_data = {}
            for name in self.entry_infos.keys():
                batch_data[name] = self.__dict__[name][indices].clone().to(self.device)
                if name in self.entry_num_classes and getattr(self, 'use_onehot_output', False):
                    batch_data[name] = one_hot(batch_data[name].long(), num_classes=self.entry_num_classes[name]).\
                        squeeze(-2).float()
            yield batch_data

    def get_recent_samples(self, num_samples) -> Dict[str, torch.Tensor]:
        assert self.size >= num_samples
        if num_samples <= self.index:
            indices = np.arange(0, num_samples)
        else:
            indices = np.concatenate([np.arange(self.size - (num_samples - self.index), self.size),
                                      np.arange(0, self.index)], axis=-1)
        batch_data = {}
        for name in self.entry_infos.keys():
            batch_data[name] = self.__dict__[name][indices].clone().to(self.device)
            if name in self.entry_num_classes and getattr(self, 'use_onehot_output', False):
                batch_data[name] = one_hot(batch_data[name].long(), num_classes=self.entry_num_classes[name]). \
                    squeeze(-2).float()
        return batch_data

    def resize(self, buffer_size):
        if buffer_size == self.buffer_size:
            return
        if buffer_size < self.buffer_size:
            logging.info('Buffer resize from {} to {}'.format(self.buffer_size, buffer_size))
            for name in self.entry_infos:
                self.__dict__[name] = self.__dict__[name][:buffer_size]
            self.size = min(self.size, buffer_size)
            self.buffer_size = buffer_size
            if self.index >= buffer_size:
                self.index = 0
            return
        if buffer_size > self.buffer_size:
            logging.info('Buffer resize from {} to {}'.format(self.buffer_size, buffer_size))
            for name in self.entry_infos:
                value = self.__dict__[name]
                self.__dict__[name] = torch.cat([value, torch.zeros(buffer_size - self.buffer_size, *value.shape[1:],
                                                                    dtype=value.dtype, device=value.device)],
                                                dim=0)
            self.buffer_size = buffer_size
            if self.index < self.size:
                self.index = self.size
            return

    def _add_offpolicy_buffer(self, buffer: SimpleUniversalBuffer):
        assert self.entry_infos == buffer.entry_infos, logging.error('to-add entries: {}\nbuffer entries: {}'.
                                                                    format(buffer.entry_infos, self.entry_infos))
        for name in self.entry_infos.keys():
            assert self.__dict__[name].shape[1:] == buffer.__dict__[name].shape[1:]
            self._insert_sequence(name, buffer.__dict__[name][:buffer.size], self.index)
        self.size = min(self.size + buffer.size, self.buffer_size)
        self.index = (self.index + buffer.size) % self.buffer_size
