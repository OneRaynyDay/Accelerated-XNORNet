"""
An example of using these layers:

cfg = {
    'VGG11': [64, 128, 128, 128, 256, 256, 256, 256, 'M', 512, 512],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, binarized=True):
        super(VGG, self).__init__()
        self.binarized = binarized
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(4*4*512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        binarized = False
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers += [nn.Dropout(0.5)]
            elif binarized:
                layers += [BinConv2dBlock(in_channels, x, kernel_size=3, padding=0, stride=1)]
                in_channels = x
            else:
                binarized = self.binarized
                layers += [Conv2dBlock(in_channels, x, kernel_size=3, padding=0, stride=1)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

VGG('VGG11') # Use the network here.

---

We get 84% test accuracy on CIFAR 10 with this model.
"""

# Torch is required for these following layers.
import torch
import torch.nn as nn
from torch.nn import functional
import math


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        We approximate the input by the following:

        input ~= sign(input) * l1_norm(input) / input.size
        """
        avg = torch.mean(torch.abs(input))
        sign = input.sign()
        ctx.save_for_backward(input, avg, sign)
        return sign * avg

    @staticmethod
    def backward(ctx, grad_output):
        """
        According to [Do-Re-Fa Networks](https://arxiv.org/pdf/1606.06160.pdf),
        the STE for binary weight networks is completely pass through.

        However, according to [Binary Neural Networks](https://arxiv.org/pdf/1602.02830.pdf),
        and [XNOR-net networks](https://arxiv.org/pdf/1603.05279.pdf),
        the STE must be thresholded by the following:

        d = d * (-1 <= w <= 1)

        Set THRESHOLD_STE to True/False for either behavior. However, it is suggested
        to set it to True because we have seen performance degradations with it = False.
        """
        sum_grad = grad_output.clone()
        input, avg, sign = ctx.saved_tensors

        # grad * avg * 1_{|xi| < 1}
        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0
        grad_output *= avg

        # Correct gradient:
        # 1/m \sum_j (grad_j * sign_j) * sign_i
        sum_grad *= sign
        sum_grad = torch.mean(sum_grad)
        grad_output += sum_grad * sign
        return grad_output


class BinarizeInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        We approximate the input by the following:

        input ~= sign(input) * l1_norm(input) / input.size
        """
        avg = torch.mean(torch.abs(input))
        sign = input.sign()
        ctx.save_for_backward(input)
        return sign * avg

    @staticmethod
    def backward(ctx, grad_output):
        """
        According to [Do-Re-Fa Networks](https://arxiv.org/pdf/1606.06160.pdf),
        the STE for binary weight networks is completely pass through.

        However, according to [Binary Neural Networks](https://arxiv.org/pdf/1602.02830.pdf),
        and [XNOR-net networks](https://arxiv.org/pdf/1603.05279.pdf),
        the STE must be thresholded by the following:

        d = d * (-1 <= w <= 1)

        Set THRESHOLD_STE to True/False for either behavior. However, it is suggested
        to set it to True because we have seen performance degradations with it = False.
        """
        input, = ctx.saved_tensors

        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0

        return grad_output


class BinarizeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        """
        We approximate the input by the following:

        input ~= sign(input) * l1_norm(input) / input.size
        """
        avg = torch.mean(torch.abs(weight))
        sign = weight.sign()
        ctx.save_for_backward(weight, avg, sign)
        return sign * avg

    @staticmethod
    def backward(ctx, grad_output):
        """
        According to [Do-Re-Fa Networks](https://arxiv.org/pdf/1606.06160.pdf),
        the STE for binary weight networks is completely pass through.

        However, according to [Binary Neural Networks](https://arxiv.org/pdf/1602.02830.pdf),
        and [XNOR-net networks](https://arxiv.org/pdf/1603.05279.pdf),
        the STE must be thresholded by the following:

        d = d * (-1 <= w <= 1)

        Set THRESHOLD_STE to True/False for either behavior. However, it is suggested
        to set it to True because we have seen performance degradations with it = False.
        """
        sum_grad = grad_output.clone()
        input, avg, sign = ctx.saved_tensors

        # grad * avg * 1_{|xi| < 1}
        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0
        grad_output *= avg

        # Correct gradient:
        # 1/m \sum_j (grad_j * sign_j) * sign_i
        sum_grad *= sign
        sum_grad = torch.mean(sum_grad)
        grad_output += sum_grad * sign
        return grad_output

    
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Takes in some inputs x, and initializes some weights for matmul,
        and performs a bitcount(xor(x, weights)).
        
        input = (N, M)
        weights = (M, K)
        
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        """
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        
        # Initializing parameters
        stdv = 1. / math.sqrt(in_features * out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        binarize_input = BinarizeInput.apply
        binarize_weight = BinarizeWeight.apply
        return functional.linear(binarize_input(input), binarize_weight(self.weight), binarize_weight(self.bias))
        

class BinConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0):
        """
        Takes in some inputs x, and initializes some weights for conv filters,
        and performs a "convolution" by binarizing the weights and multiplying
        the inputs by the binarized weights.

        input = (N, C, H, W)
        weights = (K, C, H, W) [ to be binarized ]
        biases = (K,) [ to be binarized ]
        output = (N, K, H, W)

        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

        NOTE: We skip dilation, groups, etc for now.
        """
        # TODO: Include opt-in bias parameter.
        super(BinConv2d, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, *(kernel_size, kernel_size)))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        self.stride = stride
        self.padding = padding

        # Initializing parameters
        n = in_channels
        n *= kernel_size ** 2  # number of parameters
        stdv = 1. / math.sqrt(n)

        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        binarize_input = BinarizeInput.apply
        binarize_weight = BinarizeWeight.apply
        return functional.conv2d(binarize_input(input), binarize_weight(self.weight), self.bias, self.stride,
                                 self.padding)


class BinConv2dBlock(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1):
        super(BinConv2dBlock, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.bn = nn.BatchNorm2d(input_channels)
        self.conv = BinConv2d(input_channels, output_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        weight = self.conv.weight
        self.conv.weight.data.copy_(torch.clamp(weight - torch.mean(weight), min=-1, max=1))
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
