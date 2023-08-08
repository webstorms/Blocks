import torch
import pytest

from src.snn.snn import SNN
from src.snn.block.blocks import Blocks


@pytest.fixture
def in_spikes(b=4, n=100, t=200):
    return torch.rand(b, n, t)


def get_models(n_in=100, n_out=100, rf_len=10, t_len=200, t_latency=0, recurrent=True, adapt=False):
    blocks = Blocks(n_in, n_out, rf_len, t_len, t_latency, recurrent=recurrent, adapt=adapt)
    snn = SNN(n_in, n_out, rf_len, t_len, t_latency, recurrent=recurrent, adapt=adapt)

    blocks._rf_weight = snn._rf_weight
    blocks._rf_bias = snn._rf_bias
    blocks._rec_weight = snn._rec_weight

    return blocks, snn


def test_networks_none(in_spikes):
    for t_latency in [0, 1, 2, 4, 8]:
        blocks, snn = get_models(t_latency=t_latency, recurrent=False)
        spikes1 = blocks(in_spikes, mode="train")
        spikes2 = snn(in_spikes, mode="train")

        assert torch.allclose(spikes1, spikes2)


def test_networks_recurrent(in_spikes):
    for t_latency in [0, 1, 2, 4, 8]:
        blocks, snn = get_models(t_latency=t_latency, recurrent=True)
        spikes1 = blocks(in_spikes, mode="train")
        spikes2 = snn(in_spikes, mode="train")

        assert torch.allclose(spikes1, spikes2)


def test_adaption(in_spikes):
    for t_latency in [0, 1, 2, 4, 8]:
        blocks, snn = get_models(t_latency=t_latency, recurrent=True, adapt=True)
        spikes1, mem1, x1, z1, v_th1 = blocks(in_spikes, mode="val")
        spikes2, mem2, x2, v_th2 = snn(in_spikes, mode="val")

        # v_th should equal at the start of every block
        for i in range(blocks._t_len // blocks._t_len_block):
            assert torch.allclose(v_th1[:, :, i*blocks._t_len_block], v_th2[:, :, i*blocks._t_len_block])
