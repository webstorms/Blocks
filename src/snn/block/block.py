import torch
import torch.nn as nn
import torch.nn.functional as F

from src.snn.block.util import bconv1d
from src.snn import surrogate


class Block(nn.Module):

    def __init__(self, n_in, t_len, surr_grad):
        super().__init__()
        self._n_in = n_in
        self._t_len = t_len
        self._surr_grad = surr_grad

        self._beta_ident_base = nn.Parameter(torch.ones(n_in, t_len), requires_grad=False)
        self._beta_exp = nn.Parameter(torch.arange(t_len).flip(0).unsqueeze(0).expand(n_in, t_len).float(), requires_grad=False)
        self._phi_kernel = nn.Parameter((torch.arange(t_len) + 1).flip(0).float().view(1, 1, 1, t_len), requires_grad=False)

    @staticmethod
    def g(faulty_spikes):
        negate_faulty_spikes = faulty_spikes.clone().detach()
        negate_faulty_spikes[faulty_spikes == 1.0] = 0
        faulty_spikes -= negate_faulty_spikes

        return faulty_spikes

    def forward(self, current, beta, v_init=None, v_th=1, mode="train"):

        if v_init is not None:
            current[:, :, 0] += beta * v_init

        pad_current = F.pad(current, pad=(self._t_len - 1, 0)).unsqueeze(1)

        # compute membrane potential without reset
        beta_kernel = self.build_beta_kernel(beta)
        membrane = bconv1d(pad_current, beta_kernel)

        # map no-reset membrane potentials to output spikes
        v_th = v_th.unsqueeze(1)
        faulty_spikes = surrogate.spike(membrane - v_th, self._surr_grad)

        pad_spikes = F.pad(faulty_spikes, pad=(self._t_len - 1, 0))
        z = F.conv2d(pad_spikes, self._phi_kernel)
        z_copy = z.clone().squeeze(1)

        if mode == "train":
            return Block.g(z).squeeze(1), z_copy, membrane.squeeze(1)
        elif mode == "val":
            return Block.g(z).squeeze(1), z_copy, faulty_spikes, membrane.squeeze(1)

    def build_beta_kernel(self, beta):
        beta_base = beta.unsqueeze(1).multiply(self._beta_ident_base)
        return torch.pow(beta_base, self._beta_exp).unsqueeze(1).unsqueeze(1)
