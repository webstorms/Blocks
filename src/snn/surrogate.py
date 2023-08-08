import math

import torch


class FastSigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale=10):
        ctx.scale = scale
        ctx.save_for_backward(input)

        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2
        
        return grad, None


class BoxCar(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        grad[input <= -0.5] = 0
        grad[input > 0.5] = 0

        return grad


class MG(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        lens = 0.5
        hight = 0.15
        scale = 6
        gamma = 0.5

        temp = MG.gaussian(input, mu=0., sigma=lens) * (1. + hight) - MG.gaussian(input, mu=lens, sigma=scale * lens) * hight - MG.gaussian(input, mu=-lens, sigma=scale * lens) * hight

        return gamma * grad * temp.float()

    @staticmethod
    def gaussian(x, mu=0., sigma=.5):
        return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


def spike(x, type):
    if type == "fast_sigmoid":
        return FastSigmoid.apply(x)
    elif type == "box_car":
        return BoxCar.apply(x)
    elif type == "mg":
        return MG.apply(x)
