"""Quantization"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantizer(nn.Module):
    '''
    take a real value x
    output a discrete-valued x
    '''
    def __init__(self):
        super(Quantizer, self).__init__()
        self.scale = 1

    def forward(self, inp, nbit, alpha=None, offset=None):
        self.scale = (2**nbit-1) if alpha is None else (2**nbit-1) / alpha
        if offset is None:
            out = torch.round(inp * self.scale) / self.scale
        else:
            out = (torch.round(inp * self.scale) + torch.round(offset)) / self.scale
        return out

class Signer(nn.Module):
    '''
    take a real value x
    output sign(x)
    '''
    def __init__(self):
        super(Signer, self).__init__()

    def forward(self, inp):
        return torch.sign(inp)

class ScaleSigner(nn.Module):
    '''
    take a real value x
    output sign(x) * mean(abs(x))
    '''
    def __init__(self):
        super(ScaleSigner, self).__init__()

    def forward(self, inp):
        return torch.sign(inp) * torch.mean(torch.abs(inp))

class DoReFaW(nn.Module):
    def __init__(self):
        super(DoReFaW, self).__init__()
        self.quantize = ScaleSigner()
        self.quantize2 = Quantizer()

    def forward(self, inp, nbit_w, *args, **kwargs):
        if nbit_w == 1:
            w = self.quantize(inp)
        else:
            w = torch.tanh(inp)
            maxv = torch.max(torch.abs(w))
            w = w / (2 * maxv) + 0.5
            w = 2 * self.quantize2(w, nbit_w) - 1
        return w

class DoReFaA(nn.Module):
    def __init__(self):
        super(DoReFaA, self).__init__()
        self.quantize = Quantizer()

    def forward(self, inp, nbit_a, *args, **kwargs):
        a = torch.clamp(inp, 0, 1)
        a = self.quantize(a, nbit_a, *args, **kwargs)
        return a

class PACTA(nn.Module):
    def __init__(self):
        super(PACTA, self).__init__()
        self.quantize = Quantizer()

    def forward(self, inp, nbit_a, alpha, *args, **kwargs):
        x = 0.5 * (torch.abs(inp) - torch.abs(inp-alpha) + alpha)
        return self.quantize(x, nbit_a, alpha, *args, **kwargs)

class WQuan(nn.Module):
    '''
    take a real value x
    output quantizer(x)
    '''
    def __init__(self):
        super(WQuan, self).__init__()
        self.quantizer = ScaleSigner()

    def forward(self, inp):
        return self.quantizer(inp)

class AQuan(nn.Module):
    '''
    take a real value x
    output sign(x)
    '''
    def __init__(self):
        super(AQuan, self).__init__()
        self.quantizer = Signer()

    def forward(self, inp):
        return self.quantizer(inp)

class QuanConv(nn.Conv2d):
    # general quantization for quantized conv
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w, quan_name_a, nbit_w=32, nbit_a=32,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, has_offset=False):
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': DoReFaW, 'pact': DoReFaW}
        name_a_dict = {'dorefa': DoReFaA, 'pact': PACTA}
        self.quan_w = name_w_dict[quan_name_w]()
        self.quan_a = name_a_dict[quan_name_a]()

        if quan_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.ones(1))
        else:
            self.alpha_a = None

        if quan_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.ones(1))
        else:
            self.alpha_w = None

        if has_offset:
            self.offset = nn.Parameter(torch.zeros(1))
        else:
            self.offset = None

    def forward(self, inp):
        # w quan
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x = self.quan_a(inp, self.nbit_a, self.alpha_a)
        else:
            x = inp

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DymQuanConv(nn.Conv2d):
    # dynamic quantization for quantized conv
    def __init__(self, in_channels, out_channels, kernel_size, quan_name_w, quan_name_a, nbit_w, nbit_a,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, has_offset=False):
        super(DymQuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': DoReFaW, 'pact': DoReFaW}
        name_a_dict = {'dorefa': DoReFaA, 'pact': PACTA}
        self.quan_w = name_w_dict[quan_name_w]()
        self.quan_a = name_a_dict[quan_name_a]()

        if quan_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.ones(1))
        else:
            self.alpha_a = None

        if quan_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.ones(1))
        else:
            self.alpha_w = None

        if has_offset:
            self.offset = nn.Parameter(torch.zeros(1))
        else:
            self.offset = None

    def forward(self, inp, mask):
        # w quan
        w0 = self.quan_w(self.weight, self.nbit_w-1, self.alpha_w, self.offset)
        w1 = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        w2 = self.quan_w(self.weight, self.nbit_w+1, self.alpha_w, self.offset)

        # a quan
        x0 = self.quan_a(inp, self.nbit_a-1, self.alpha_a)
        x1 = self.quan_a(inp, self.nbit_a, self.alpha_a)
        x2 = self.quan_a(inp, self.nbit_a+1, self.alpha_a)

        x0 = F.conv2d(x0, w0, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x1 = F.conv2d(x1, w1, self.bias, self.stride, self.padding, self.dilation, self.groups)
        x2 = F.conv2d(x2, w2, self.bias, self.stride, self.padding, self.dilation, self.groups)

        x = x0 * mask[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3) + \
            x1 * mask[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3) + \
            x2 * mask[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        return x
