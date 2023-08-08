import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainbox.models import BBModel

from src.snn import surrogate

# SNN control models.


class BaseSNN(BBModel):

    MIN_BETA = 0.001
    MAX_BETA = 0.999

    def __init__(self, n_in, n_out, rf_len, t_len, t_latency, recurrent=True, beta_grad=True, adapt=True, init_beta=1, init_p=1, detach_spike_grad=True, surr_grad="fast_sigmoid"):
        super().__init__()
        self._n_in = n_in
        self._n_out = n_out
        self._rf_len = rf_len
        self._t_len = t_len
        self._t_latency = t_latency
        self._recurrent = recurrent
        self._beta_grad = beta_grad
        self._adapt = adapt
        self._detach_spike_grad = detach_spike_grad
        self._surr_grad = surr_grad

        self._beta = nn.Parameter(data=torch.Tensor(n_out * [init_beta]), requires_grad=beta_grad)
        self._rf_weight = nn.Parameter(torch.rand(n_out, 1, n_in, self._rf_len), requires_grad=True)
        self._rf_bias = nn.Parameter(torch.zeros(n_out), requires_grad=True)

        self._rec_weight = nn.Parameter(torch.rand(n_out, n_out), requires_grad=recurrent)

        self._p = nn.Parameter(data=torch.Tensor(n_out * [init_p]), requires_grad=adapt)
        self._b = nn.Parameter(data=torch.Tensor(n_out * [1.8]), requires_grad=adapt)

        self.init_weight(self._rf_weight, "uniform", a=-1 / np.sqrt(n_in * rf_len), b=1 / np.sqrt(n_in * rf_len))
        self.init_weight(self._rec_weight, "identity")

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out, "rf_len": self._rf_len, "t_len": self._t_len, "t_latency": self._t_latency, "recurrent": self._recurrent, "beta_grad": self._beta_grad, "adapt": self._adapt, "detach_spike_grad": self._detach_spike_grad, "surr_grad": self._surr_grad}

    @property
    def p(self):
        return torch.clamp(self._p.abs(), min=0, max=0.999)

    @property
    def b(self):
        return torch.clamp(self._b.abs(), min=0.001, max=1)

    @property
    def beta(self):
        return torch.clamp(self._beta, min=BaseSNN.MIN_BETA, max=BaseSNN.MAX_BETA)

    @property
    def rec_weight(self):
        return self._rec_weight

    def get_rec_input(self, spikes):
        return torch.einsum("ij, bj...->bi...", self.rec_weight, spikes.detach() if self._detach_spike_grad else spikes)

    def forward(self, x, mode="train"):
        # x: b x n x t

        x = F.pad(x, (self._rf_len - 1, 0))
        x = x.unsqueeze(1)  # Add channel dim
        x = F.conv2d(x, self._rf_weight, self._rf_bias)[:, :, 0]  # Slice out height dim

        return self.process(x, mode)

    def process(self, x, mode):
        raise NotImplementedError


class SNN(BaseSNN):

    def __init__(self, n_in, n_out, rf_len, t_len, t_latency, recurrent=False, beta_grad=True, adapt=True, init_beta=1, init_p=1, detach_spike_grad=True, surr_grad="fast_sigmoid"):
        super().__init__(n_in, n_out, rf_len, t_len, t_latency, recurrent, beta_grad, adapt, init_beta, init_p, detach_spike_grad, surr_grad)

    def process(self, x, mode="train"):
        # x: b x n x t

        mem_list = []
        spikes_list = []
        spikes = torch.zeros_like(x).to(x.device)[:, :, 0]
        rec_current = torch.zeros_like(x)
        mem = torch.zeros_like(x).to(x.device)[:, :, 0]
        refac_times = torch.zeros_like(x).to(x.device)[:, :, 0] + self._t_latency

        v_th = torch.ones_like(x).to(x.device)[:, :, 0]
        a = torch.zeros_like(x).to(x.device)[:, :, 0]
        v_th_list = []

        for t in range(x.shape[2]):
            stimulus_current = x[:, :, t]
            rec_current[:, :, t] = self.get_rec_input(spikes)

            # Recurrent latency
            if t >= self._t_latency and self._recurrent:
                input_current = stimulus_current + rec_current[:, :, t-self._t_latency]
            else:
                input_current = stimulus_current

            # Apply absolute refractory period
            refac_times[spikes > 0] = 0
            refac_mask = refac_times < self._t_latency
            input_current[refac_mask] = 0
            refac_times += 1

            new_mem = torch.einsum("bn...,n->bn...", mem, self.beta) + input_current
            spikes = surrogate.spike(new_mem - v_th, self._surr_grad)

            mem_list.append(new_mem)
            if self._detach_spike_grad:
                mem = new_mem * (1 - spikes.detach())
            else:
                mem = new_mem * (1 - spikes)
            # new_mem -= new_mem * spikes (should be same as above?)
            spikes_list.append(spikes)

            if self._adapt:
                a = self.p * a + spikes
                v_th = 1 + self.b * a
            v_th_list.append(v_th)

        if mode == "train":
            return torch.stack(spikes_list, dim=2)
        elif mode == "val":
            v_th = torch.stack(v_th_list, dim=2)
            v_th = torch.roll(v_th, 1, dims=2)
            v_th[:, :, :1] = 1

            return torch.stack(spikes_list, dim=2), torch.stack(mem_list, dim=2), x, v_th


class SNNIntegrator(SNN):

    def __init__(self, n_in, n_out, t_len, init_beta=1):
        super().__init__(n_in, n_out, 1, t_len, t_latency=0, recurrent=False, beta_grad=True, adapt=False, init_beta=init_beta, init_p=1, detach_spike_grad=True, surr_grad="fast_sigmoid")

    def process(self, x, mode="train"):
        mem_list = []
        mem = torch.zeros_like(x).to(x.device)[:, :, 0]

        for t in range(x.shape[2]):
            input_current = x[:, :, t]

            new_mem = torch.einsum("bn...,n->bn...", mem, self.beta) + input_current
            mem_list.append(new_mem)
            mem = new_mem

        return torch.stack(mem_list, dim=2)
