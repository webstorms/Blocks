import torch
import pytest

from src.snn.block.block import Block


@pytest.fixture
def block():
    return Block(2, 4, "fast_sigmoid")


def test_beta_kernel(block):
    # Use a single beta
    assert torch.allclose(block.build_beta_kernel(torch.Tensor([0.1]))[0, 0, 0], torch.Tensor([0.0010, 0.0100, 0.1000, 1.0000]))
    assert torch.allclose(block.build_beta_kernel(torch.Tensor([0.1]))[1, 0, 0], torch.Tensor([0.0010, 0.0100, 0.1000, 1.0000]))

    # Use multiple beta
    assert torch.allclose(block.build_beta_kernel(torch.Tensor([0.1]))[0, 0, 0], torch.Tensor([0.0010, 0.0100, 0.1000, 1.0000]))
    assert torch.allclose(block.build_beta_kernel(torch.Tensor([0.5]))[1, 0, 0], torch.Tensor([0.1250, 0.2500, 0.5000, 1.0000]))


def test_phi_kernel(block):
    assert torch.allclose(block._phi_kernel, torch.Tensor([[[[4., 3., 2., 1.]]]]))


def test_g(block):
    phi_spikes = torch.zeros(2, 4)
    phi_spikes[0, 0] = 1
    phi_spikes[0, 2] = 2
    phi_spikes[1, 1] = 1
    phi_spikes[1, 3] = 3
    assert block.g(phi_spikes).sum() == 2


def test_differentiable_vars(block):
    assert not block._beta_ident_base.requires_grad
    assert not block._beta_exp.requires_grad
    assert not block._phi_kernel.requires_grad