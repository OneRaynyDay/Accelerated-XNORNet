# Torch is required for these following layers.
import torch
import torch.nn as nn
from torch.nn import functional
import math

class BinarizeInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        We approximate the input by the following:

        input ~= sign(input) * l1_norm(input) / input.size
        """
        # avg = torch.mean(torch.abs(input))
        sign = input.sign()
        ctx.save_for_backward(input)
        return sign # * avg

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
        # avg = torch.mean(torch.abs(weight))
        sign = weight.sign()
        ctx.save_for_backward(weight, sign) #avg, sign)
        return sign # * avg

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
        input, sign = ctx.saved_tensors

        # grad * avg * 1_{|xi| < 1}
        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0
        
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
        self.bias = None # disable bias
        # self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        
        # Initializing parameters
        stdv = 1. / math.sqrt(in_features * out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        binarize_input = BinarizeInput.apply
        binarize_weight = BinarizeWeight.apply
        return functional.linear(binarize_input(input), binarize_weight(self.weight))