import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=3, eps=1e-6):
    """
    Apply Generalized Mean Pooling (GeM) to a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        p (float): The p-value for the generalized mean. Default is 3.
        eps (float): Added to the denominator to prevent division by zero. Default is 1e-6.

    Returns:
        torch.Tensor: GeM-pooled representation of the input tensor.
    """
    x_shape = [int(s) for s in x.shape[2:]]
    return F.avg_pool2d(x.clamp(min=eps).pow(p), x_shape).pow(1.0 / p)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer for global average pooling.

    Attributes:
        p (float or torch.Tensor): The p-value for the generalized mean.
        eps (float): A small constant added to the denominator to prevent division by zero.
    """

    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        """
        Initialize the GeM layer.

        Args:
            p (float or torch.Tensor): The p-value for the generalized mean.
            eps (float, optional): Eps to prevent division by zero. Defaults to 1e-6.
            p_trainable (bool, optional): Whether p is trainable. Defaults to False.
        """
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        """
        Perform the GeM pooling operation on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: GeM-pooled representation of the input tensor.
        """
        ret = gem(x, p=self.p, eps=self.eps)
        return ret
