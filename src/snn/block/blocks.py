import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.snn.snn import BaseSNN
from src.snn.block.block import Block
from src.snn.block.util import time_cat, bconv1d


class Blocks(BaseSNN):

    def __init__(self, n_in, n_out, rf_len, t_len, t_latency, recurrent=True, beta_grad=True, adapt=True, init_beta=1, init_p=1, detach_spike_grad=True, surr_grad="fast_sigmoid"):
        super().__init__(n_in, n_out, rf_len, t_len, t_latency, recurrent, beta_grad, adapt, init_beta, init_p, detach_spike_grad, surr_grad)

        self._t_len_block = t_latency + 1
        self._block = Block(n_out, self._t_len_block, surr_grad)
        self._n_blocks = math.ceil(t_len / self._t_len_block)
        self._t_pad = self._n_blocks * self._t_len_block - self._t_len

        self._p_ident_base = nn.Parameter(torch.ones(n_out, self._t_len_block), requires_grad=False)
        self._p_exp = nn.Parameter(torch.arange(1, self._t_len_block + 1).float(), requires_grad=False)

    def process(self, x, mode="train"):
        x_init = x
        if self._t_pad != 0:
            x = F.pad(x, pad=(0, self._t_pad))

        mem_list = []
        spikes_list = []
        z_list = []

        z = torch.zeros_like(x[:, :, self._t_len_block:])
        v_init = torch.zeros_like(x[:, :, 0]).to(x.device)
        int_mem = torch.zeros_like(x[:, :, 0]).to(x.device)

        a_kernel = torch.zeros_like(x).to(x.device)[:, :, :self._t_len_block]
        v_th = torch.ones_like(x).to(x.device)[:, :, :self._t_len_block]
        v_th_list = []

        for i in range(self._n_blocks):
            x_slice = x[:, :, i * self._t_len_block: (i+1) * self._t_len_block]

            # Recurrent current and refractory mask only included after first block
            if i > 0:
                # Add recurrent current to input
                if self._recurrent:
                    rec_current = self.get_rec_input(spikes)
                    x_slice = x_slice + rec_current

                # Apply refractory mask to input
                if self._detach_spike_grad:
                    spike_mask = spikes.detach().amax(dim=2).bool()
                else:
                    spike_mask = spikes.amax(dim=2).bool()
                refac_mask = (z < spike_mask.unsqueeze(2)) * x_slice
                x_slice -= refac_mask

                # Set initial membrane potentials
                v_init = int_mem[:, :, -1] * ~spike_mask  # if spiked -> zero initial membrane potential

                # Set initial adaptive params
                if self._adapt:
                    # Get a at time of spike + spike (which is equal to 1/p to account for raising v_th by 1 next step
                    # do the math or see paper if this is not clear)
                    if self._detach_spike_grad:
                        a_at_spike = (a_kernel * spikes.detach()).sum(dim=2) + (1 / self.p)
                    else:
                        a_at_spike = (a_kernel * spikes).sum(dim=2) + (1 / self.p)
                    decay_steps = (z > 1).sum(dim=2)  # Compute number of decay steps
                    new_a = a_at_spike * torch.pow(self.p.unsqueeze(0), decay_steps)
                    a = (a_kernel[:, :, -1] * ~spike_mask) + (new_a * spike_mask)

                    # Update a for neurons that spiked
                    a_kernel = self.compute_a_kernel(a, self.p)
                    v_th = 1 + self.b.view(1, -1, 1) * a_kernel

            if mode == "train":
                spikes, z, int_mem = self._block(x_slice, self.beta, v_init=v_init, v_th=v_th, mode="train")
                spikes_list.append(spikes)
            elif mode == "val":
                spikes, z, _, int_mem = self._block(x_slice, self.beta, v_init=v_init, v_th=v_th, mode="val")
                spikes_list.append(spikes)
                mem_list.append(int_mem)
                z_list.append(z)
                v_th_list.append(v_th)

        if mode == "train":
            return time_cat(spikes_list, self._t_pad)
        elif mode == "val":
            return time_cat(spikes_list, self._t_pad), time_cat(mem_list, self._t_pad), x_init, time_cat(z_list, self._t_pad), time_cat(v_th_list, self._t_pad)

    def compute_a_kernel(self, a, p):
        # a: b x n
        # p: n
        # output: b x n x t

        return torch.pow(p.unsqueeze(-1) * self._p_ident_base, self._p_exp).unsqueeze(0) * a.unsqueeze(-1)


class BlocksIntegrator(BaseSNN):

    def __init__(self, n_in, n_out, t_len, init_beta=1):
        super().__init__(n_in, n_out, 1, t_len, t_latency=0, recurrent=False, beta_grad=True, adapt=False, init_beta=init_beta, init_p=1, detach_spike_grad=True, surr_grad="fast_sigmoid")
        self._block = Block(n_out, t_len, "fast_sigmoid")

    def process(self, x, mode="train"):
        pad_current = F.pad(x, pad=(self._t_len - 1, 0)).unsqueeze(1)

        # compute membrane potential without reset
        beta_kernel = self._block.build_beta_kernel(self.beta)
        membrane = bconv1d(pad_current, beta_kernel)

        return membrane.squeeze(1)
