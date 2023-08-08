import torch
from torch.distributions.poisson import Poisson
from brainbox.datasets import BBDataset


class SyntheticSpikes(BBDataset):
    
    """
    This is the synthetic spike dataset with which the model was benchmarked.
    """

    def __init__(self, t_len, n_units, min_r, max_r, n_samples):
        super().__init__(None)
        self.t_len = t_len
        self.n_units = n_units
        self.min_r = min_r
        self.max_r = max_r
        self.n_samples = n_samples

    def __getitem__(self, i):
        rate = torch.FloatTensor(1).uniform_(self.min_r, self.max_r).item()
        x = self._create_spikes(rate, self.n_units, self.t_len)

        return x

    def __len__(self):
        return self.n_samples

    def _load_dataset(self, train):
        return None, None

    def _create_spikes(self, rate, n_units, t_len):
        pois_dis = Poisson(rate/t_len)
        if type(n_units) == tuple:
            samples = pois_dis.sample(sample_shape=(*n_units, t_len))
        else:
            samples = pois_dis.sample(sample_shape=(n_units, t_len))
        samples[samples > 1] = 1

        return samples