import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainbox import models

import src.snn.block.blocks as blocks
import src.snn.snn as snn


class AuditoryModel(models.BBModel):

    # AuditoryModel name comes from the SHD dataset being auditory

    HIDDEN_MEM_TIME = 20
    HIDDEN_ADAPT_TIME = 150
    READOUT_MEM_TIME = 20

    def __init__(self, method, n_in, n_hidden, n_out, t_len, abs_refac, recurrent=True, dt=1, surr_grad="fast_sigmoid", detach_spike_grad=True):
        super().__init__()
        self._method = method
        self._n_in = n_in
        self._n_hidden = n_hidden
        self._n_out = n_out
        self._t_len = t_len
        self._abs_refac = abs_refac
        self._recurrent = recurrent
        self._dt = dt
        self._surr_grad = surr_grad

        init_hidden_beta = np.exp(-dt / AuditoryModel.HIDDEN_MEM_TIME)
        init_hidden_p = np.exp(-dt / AuditoryModel.HIDDEN_ADAPT_TIME)
        init_readout_beta = np.exp(-dt / AuditoryModel.READOUT_MEM_TIME)

        if method == "standard":
            self._thalamic_layer = snn.SNN(n_in, n_hidden, 1, t_len, abs_refac, recurrent, beta_grad=True, adapt=True, init_beta=init_hidden_beta, init_p=init_hidden_p, surr_grad=self._surr_grad)
            self._cortical_layer = snn.SNN(n_hidden, n_hidden, 1, t_len, abs_refac, recurrent, beta_grad=True, adapt=True, init_beta=init_hidden_beta, init_p=init_hidden_p, surr_grad=self._surr_grad)
            self._output = snn.SNNIntegrator(n_hidden, n_out, t_len, init_beta=init_readout_beta)
        else:
            self._thalamic_layer = blocks.Blocks(n_in, n_hidden, 1, t_len, abs_refac, recurrent, beta_grad=True, adapt=True, init_beta=init_hidden_beta, init_p=init_hidden_p, surr_grad=self._surr_grad, detach_spike_grad=detach_spike_grad)
            self._cortical_layer = blocks.Blocks(n_hidden, n_hidden, 1, t_len, abs_refac, recurrent, beta_grad=True, adapt=True, init_beta=init_hidden_beta, init_p=init_hidden_p, surr_grad=self._surr_grad, detach_spike_grad=detach_spike_grad)
            self._output = blocks.BlocksIntegrator(n_hidden, n_out, t_len, init_beta=init_readout_beta)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "method": self._method, "n_in": self._n_in, "n_hidden": self._n_hidden, "n_out": self._n_out, "t_len": self._t_len, "abs_refac": self._abs_refac, "recurrent": self._recurrent, "dt": self._dt, "surr_grad": self._surr_grad}

    def forward(self, x, mode="train"):
        # x: b x n x t
        thalamic_output = self._thalamic_layer(x, mode)
        cortical_output = self._cortical_layer(thalamic_output if mode == "train" else thalamic_output[0], mode)

        if mode == "train":
            return self._output(cortical_output, mode).sum(2)
        else:
            return self._output(cortical_output[0], mode).sum(2), cortical_output, thalamic_output


class ModelBuilder(models.BBModel):

    def __init__(self, method, t_len, abs_refac, n_in, n_hidden, n_layers):
        super().__init__()
        self._layers = nn.ModuleList()

        for i in range(n_layers):
            n_in = n_in if i == 0 else n_hidden
            if method == "standard":
                self._layers.append(snn.SNN(n_in, n_hidden, 1, t_len, abs_refac, recurrent=True, beta_grad=True, adapt=True, init_beta=0.9, init_p=0.9, surr_grad="mg"))
            else:
                self._layers.append(blocks.Blocks(n_in, n_hidden, 1, t_len, abs_refac, recurrent=True, beta_grad=True, adapt=True, init_beta=0.9, init_p=0.9, surr_grad="mg"))

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)

        return x


class Neuron(models.BBModel):

    def __init__(self, method, abs_refac_ms, downsample=1, dt01ref=False):
        super().__init__()
        self.method = method
        self.abs_refac_ms = abs_refac_ms
        self.downsample = downsample
        self.dt01ref = dt01ref
        self.dt_ms = downsample * 0.1  # Larger dt_ms == downsample temporal resolution

        if not dt01ref:
            self.neuron = Neuron.get_neuron(method, abs_refac_ms, self.dt_ms, downsample)
        else:
            self.neuron = Neuron.get_neuron(method, abs_refac_ms, 0.5*self.dt_ms, 0.5*downsample)
        upsample_kernel = torch.zeros(downsample)
        upsample_kernel[0] = 1
        self.upsample_kernel = nn.Parameter(upsample_kernel.view(1, 1, -1), requires_grad=False)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "method": self.method, "abs_refac_ms": self.abs_refac_ms, "downsample": self.downsample, "dt01ref": self.dt01ref}

    def forward(self, x, mode="train"):
        x = F.avg_pool1d(x, self.downsample, self.downsample)  # Down sample the signal (if need be)
        spikes = self.neuron(x, mode)

        # Return data for plotting
        if mode == "val":
            spikes, mem = spikes[0], spikes[1]

            return spikes, mem

        # Return predicted spike train for training
        if self.dt01ref:  # When running in DT=0.05ms (was added on to run additional experiments for a reviewer)
            spikes = F.max_pool1d(spikes, 2, 2)
            return spikes

        if self.downsample == 1:
            return spikes
        else:
            return F.conv_transpose1d(spikes, self.upsample_kernel, stride=self.downsample)

    @staticmethod
    def get_neuron(method, abs_refac_ms, dt_ms, downsample):
        init_beta = np.exp(-dt_ms / 20)
        init_p = np.exp(-dt_ms / 100)
        t_len = int(1000 / dt_ms)
        abs_refac_ms = int(abs_refac_ms / dt_ms)

        if method == "blocks":
            neuron = blocks.Blocks(1, 1, rf_len=1, t_len=t_len, t_latency=abs_refac_ms, recurrent=False, beta_grad=True, adapt=True, init_beta=init_beta, init_p=init_p, detach_spike_grad=True, surr_grad="mg")
        else:
            neuron = snn.SNN(1, 1, rf_len=1, t_len=t_len, t_latency=abs_refac_ms, recurrent=False, beta_grad=True, adapt=True, init_beta=init_beta, init_p=init_p, detach_spike_grad=True, surr_grad="mg")

        neuron.init_weight(neuron._rf_weight, "constant", c=downsample)
        neuron._b = nn.Parameter(data=torch.Tensor([0.1 / downsample]), requires_grad=True)

        return neuron
