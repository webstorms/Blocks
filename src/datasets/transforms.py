import torch
import numpy as np
import brainbox
from brainbox.datasets.transforms import BBTransform


class SpikeTensorBuilder(BBTransform):

    def __init__(self, n_units, t_len, dt):
        self._n_units = n_units
        self._t_len = t_len
        self._dt = dt

    def __call__(self, args):
        units, times = args[0], args[1]
        units = units % self._n_units
        times = torch.round(times * 1000. / self._dt).int()

        # Constrain spike length
        idxs = (times < self._t_len)
        units = units[idxs]
        times = times[idxs]

        # Build COO tensor
        indices = torch.stack([torch.Tensor(units.tolist()), torch.Tensor(times.tolist())], dim=0).long()
        shape = torch.Size([self._n_units, self._t_len, ])
        spikes = torch.FloatTensor(np.ones(len(indices[0])))

        return torch.sparse.FloatTensor(indices, spikes, shape).to_dense()


class List:

    @staticmethod
    def get_nmnist_transform(t_len, use_augmentation=False):
        if use_augmentation:
            raise NotImplementedError
        else:
            transform_list = [SpikeTensorBuilder(n_units=1156, t_len=t_len, dt=1)]

        return brainbox.datasets.transforms.Compose(transform_list)

    @staticmethod
    def get_shd_transform(t_len):
        transform_list = [SpikeTensorBuilder(n_units=700, t_len=t_len, dt=2)]

        return brainbox.datasets.transforms.Compose(transform_list)
