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


def gem_freq(x, p=3, eps=1e-6, exportable=False):
    if exportable:
        return x.clamp(min=eps).pow(p).mean(2, keepdims=True).pow(1.0 / p)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6, exportable=False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.exportable = exportable

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps, exportable=self.exportable)


class GeMFreqFixed(nn.Module):
    def __init__(self, time_kernel_size, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, time_kernel_size))

    def forward(self, x):
        return self.avg_pool(
            x.clamp(min=self.eps).pow(self.p).mean(2, keepdims=True)
        ).pow(1.0 / self.p)


class Attention(nn.Module):
    """
    Attention module for sequence data.

    Attributes:
        hidden_dim (int): The dimension of the input sequence.
        attention_dim (int): The dimension of the attention layer.
    """

    def __init__(self, hidden_dim, attention_dim=None):
        """
        Constructor

        Args:
            hidden_dim (int): The dimension of the input sequence.
            attention_dim (int, optional): The dimension of the attention layer.
                Defaults to None, in which case it's set to `hidden_dim`.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        if self.attention_dim is None:
            self.attention_dim = self.hidden_dim
        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, x):
        """
        Perform the forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input sequence data of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Attention-weighted representation of the input sequence.
        """
        batch_size, seq_len, _ = x.size()
        H = torch.tanh(self.proj_w(x))
        att_scores = torch.softmax(self.proj_v(H), axis=1)
        attn_x = (x * att_scores).sum(1)
        return attn_x


class FreqAttention(nn.Module):
    def __init__(
        self,
        in_chanels,
        hidden_chanels=512,
        p=0.,
        num_classes=182,
        exportable=False,
    ):
        super().__init__()
        self.pooling = GeMFreq(exportable=exportable)

        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chanels, hidden_chanels),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels=hidden_chanels,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels=hidden_chanels,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat):
        feat = self.pooling(feat)
        feat = feat.squeeze(-2).permute(0, 2, 1)  # (bs, time, ch)
        feat = self.dense_layers(feat).permute(0, 2, 1)  # (bs, 512, time)

        time_att = torch.tanh(self.attention(feat))
        feat_v = self.fix_scale(feat)

        logits = torch.sum(
            feat_v * torch.softmax(time_att, dim=-1),
            dim=-1,
        )
        return logits
