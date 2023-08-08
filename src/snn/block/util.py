import torch
import torch.nn.functional as F


def bconv1d(x, weight, stride=1, dilation=1, padding=0):
    # Would be useful if PyTorch provided batched 1D convs in their library
    b, c, n, h = x.shape
    n, out_channels, in_channels, kernel_width_size = weight.shape

    out = x.view(b, c * n, h)
    weight = weight.view(n * out_channels, in_channels, kernel_width_size)

    out = F.conv1d(out, weight=weight, bias=None, stride=stride, dilation=dilation, groups=n, padding=padding)

    return out.view(b, c, n, -1)


def time_cat(tensor_list, t_pad):
    tensor = torch.cat(tensor_list, dim=2)

    if t_pad > 0:
        tensor = tensor[:, :, :-t_pad]

    return tensor
