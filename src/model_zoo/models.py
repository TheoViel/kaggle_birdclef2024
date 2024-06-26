import timm
import torch
import warnings
import torch.nn as nn

from model_zoo.layers import GeM
from util.torch import load_model_weights
from model_zoo.melspec import FeatureExtractor

warnings.simplefilter(action="ignore", category=UserWarning)


def define_model(
    name,
    melspec_params,
    head="gem",
    aug_config=None,
    num_classes=182,
    n_channels=1,
    drop_rate=0,
    drop_path_rate=0,
    pretrained_weights="",
    pretrained=True,
    reduce_stride=False,
    norm="min_max",
    top_db=None,
    verbose=1,
):
    """
    Defines the model.

    Args:
        name (str): Name of the model architecture.
        melspec_params (dict): Parameters for Mel spectrogram creation.
        head (str, optional): Type of head layer. Defaults to "gem".
        aug_config (dict, optional): Configuration for data augmentation. Defaults to None.
        num_classes (int, optional): Number of output classes. Defaults to 182.
        n_channels (int, optional): Number of input channels. Defaults to 1.
        drop_rate (float, optional): Dropout rate. Defaults to 0.
        drop_path_rate (float, optional): Drop path rate. Defaults to 0.
        pretrained_weights (str, optional): Path to pretrained weights. Defaults to "".
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        reduce_stride (bool, optional): Whether to reduce stride. Defaults to False.
        norm (str, optional): Type of normalization. Defaults to "min_max".
        top_db (float, optional): Top decibel threshold. Defaults to None.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        ClsModel: Defined classification model.
    """
    if drop_path_rate > 0 and "coat" not in name and "vit" not in name and "ghost" not in name:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate if head == "" else 0,
            global_pool="" if head == "gem" else None,
            num_classes=num_classes if head == "" else 0,
        )
    else:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            drop_rate=drop_rate if head == "" else 0,
            global_pool="" if head == "gem" else None,
            num_classes=num_classes if head == "" else 0,
        )
    encoder.name = name

    ft_extractor = FeatureExtractor(
        melspec_params,
        aug_config=aug_config,
        norm=norm,
        top_db=top_db,
    )

    model = ClsModel(
        encoder,
        ft_extractor,
        num_classes=num_classes,
        n_channels=n_channels,
        drop_rate=drop_rate,
        head=head,
    )

    if pretrained_weights:
        model = load_model_weights(
            model, pretrained_weights, verbose=verbose, strict=False
        )

    if reduce_stride:
        model.reduce_stride()

    return model


class ClsModel(nn.Module):
    """
    PyTorch model for wave classification.
    """

    def __init__(
        self,
        encoder,
        ft_extractor,
        num_classes=2,
        n_channels=3,
        drop_rate=0,
        head="gem",
    ):
        """
        Constructor for the classification model.

        Args:
            encoder (torch model): The feature encoder.
            ft_extractor (FeatureExtractor): The feature extractor.
            num_classes (int): The number of primary classes.
            n_channels (int): The number of input channels.
            drop_rate (float): Dropout rate.
        """
        super().__init__()
        self.encoder = encoder
        self.ft_extractor = ft_extractor

        self.nb_ft = encoder.num_features
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.use_gem = head == "gem"
        self.head = head

        if head == "gem":
            self.global_pool = GeM(p_trainable=False) if self.use_gem else None
            self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()
            self.logits = nn.Linear(self.nb_ft, num_classes)
        else:
            self.logits = None

        self._update_num_channels()

    def _update_num_channels(self):
        """
        Update the number of input channels for the encoder.
        """
        if self.n_channels != 3:
            if "convnext" in self.encoder.name:
                conv = self.encoder.stem[0]
            elif "coat_lite" in self.encoder.name:
                conv = self.encoder.patch_embed1.proj
            elif (
                "coatnet" in self.encoder.name
                or "nfnet" in self.encoder.name
                or "maxvit" in self.encoder.name
            ):
                conv = self.encoder.stem.conv1
            elif "efficientnet" in self.encoder.name:
                conv = self.encoder.conv_stem
            elif "efficientvit" in self.encoder.name:
                conv = self.encoder.stem.in_conv.conv
            else:
                raise NotImplementedError

            new_conv = nn.Conv2d(
                self.n_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
            )

            new_conv_w = new_conv.weight.clone().detach()

            if self.n_channels > 3:
                new_conv_w[:, :3] = conv.weight.clone().detach()
            elif self.n_channels == 2:
                new_conv_w = conv.weight.clone().detach()[:, :2]
            elif self.n_channels == 1:
                new_conv_w = conv.weight.clone().detach().mean(1, keepdims=True)

            new_conv.weight = torch.nn.Parameter(new_conv_w, requires_grad=True)

            if conv.bias is not None:
                new_conv_b = conv.bias.clone().detach()
                new_conv.bias = torch.nn.Parameter(new_conv_b, requires_grad=True)

            if "convnext" in self.encoder.name:
                self.encoder.stem[0] = new_conv
            elif "coat_lite" in self.encoder.name:
                self.encoder.patch_embed1.proj = new_conv
            elif (
                "coatnet" in self.encoder.name
                or "nfnet" in self.encoder.name
                or "maxvit" in self.encoder.name
            ):
                self.encoder.stem.conv1 = new_conv
            elif "efficientnet" in self.encoder.name:
                self.encoder.conv_stem = new_conv
            elif "efficientvit" in self.encoder.name:
                self.encoder.stem.in_conv.conv = new_conv

    def reduce_stride(self):
        """
        Reduce the stride of the first layer of the encoder.
        """
        if "nfnet" in self.encoder.name:
            self.encoder.stem_conv1.stride = (1, 1)
        elif "efficientnet" in self.encoder.name:
            self.encoder.conv_stem.stride = (1, 1)
        elif "resnet" in self.encoder.name or "resnext" in self.encoder.name:
            try:
                self.encoder.conv1[0].stride = (1, 1)
            except Exception:
                self.encoder.conv1.stride = (1, 1)
        elif "maxvit" in self.encoder.name or "maxxvit" in self.encoder.name:
            self.encoder.stem.conv1.stride = (1, 1)
        elif "coatnet" in self.encoder.name:
            self.encoder.stem.conv1.stride = (1, 1)
        else:
            raise NotImplementedError

    def get_logits(self, fts):
        """
        Compute logits for the primary and auxiliary classes.

        Args:
            fts (torch.Tensor): Features of shape [batch_size x num_features].

        Returns:
            torch.Tensor: Logits for the primary classes of shape [batch_size x num_classes].
        """
        if self.use_gem:
            fts = self.global_pool(fts)[:, :, 0, 0]
        else:  # Avg pooling
            while len(fts.size()) > 2:
                fts = fts.mean(-1)

        fts = self.dropout(fts)
        return self.logits(fts)

    def forward(self, x, y=None, y_aux=None, w=None):
        """
        Forward function for the model.

        Args:
            x (torch.Tensor): Input waves of shape [batch_size x max_len].
            y (torch.Tensor, optional): Target labels. Defaults to None.
            y_aux (torch.Tensor, optional): Auxiliary labels. Defaults to None.
            w (torch.Tensor, optional): Auxiliary weights. Defaults to None.

        Returns:
            torch.Tensor: Logits for the primary classes of shape [batch_size x num_classes].
            torch.Tensor: Target labels, updated with mixup if mixup is applied.
            torch.Tensor: Auxiliary labels, updated with mixup if mixup is applied.
            torch.Tensor: Auxiliary weights, updated with mixup if mixup is applied.
        """
        n_chunks = x.size(1) // (32000 * 5)

        melspec, y, y_aux, w = self.ft_extractor(x, y, y_aux=y_aux, w=w)

        if n_chunks > 1:  # bs x n_mels x t -> bs * n_chunks x n_mels x t / n_chunks
            bs, n_mels, t = melspec.size()
            t = t // n_chunks * n_chunks
            melspec = melspec[:, :, :t]
            melspec = melspec.reshape(bs, n_mels, n_chunks, t // n_chunks)
            melspec = melspec.permute(0, 2, 1, 3)
            melspec = melspec.reshape(bs * n_chunks, n_mels, t // n_chunks)

        melspec = melspec.unsqueeze(1)
        if self.n_channels == 3:
            pos = torch.linspace(0., 1., melspec.size(2)).to(melspec.device).view(1, 1, -1, 1)
            pos = pos.expand(melspec.size(0), 1, melspec.size(2), melspec.size(3))
            melspec = torch.cat([melspec, melspec, pos], 1)

        fts = self.encoder(melspec)

        if (
            n_chunks > 1
        ):  # bs * n_chunks x n_fts x f x t / n_chunks -> bs x n_fts x f x t
            _, nb_ft, f, t_c = fts.size()
            fts = fts.reshape(bs, n_chunks, nb_ft, f, t_c)
            fts = fts.permute(0, 2, 3, 1, 4)
            fts = fts.reshape(bs, nb_ft, f, n_chunks * t_c)

        logits = self.get_logits(fts) if self.logits is not None else fts
        return logits, y, y_aux, w
