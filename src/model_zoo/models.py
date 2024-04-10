import timm
import torch
import warnings
import torch.nn as nn

from model_zoo.layers import GeM, FreqAttention
from util.torch import load_model_weights
from model_zoo.melspec import FeatureExtractor

warnings.simplefilter(action="ignore", category=UserWarning)


def define_model(
    name,
    melspec_params,
    head="freq_att",
    spec_augment_config=None,
    num_classes=182,
    n_channels=1,
    drop_rate=0,
    drop_path_rate=0,
    pretrained_weights="",
    pretrained=True,
    increase_stride=False,
    replace_pad_conv=False,
    verbose=1,
):
    """

    """
    if drop_path_rate > 0 and "coat" not in name:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            global_pool="avg" if "coat" in name else "",
        )
    else:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg" if "coat" in name else "",
        )
    encoder.name = name

    ft_extractor = FeatureExtractor(melspec_params, spec_augment_config)

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

    if increase_stride:
        model.increase_stride()

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
            encoder: The feature encoder.
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

        if head == "freq_att":
            self.freq_att = FreqAttention(
                in_chans=self.nb_ft,
                # exportable=False,
                num_classes=num_classes,
                p=drop_rate,
            )
        else:
            self.global_pool = GeM(p_trainable=False) if self.use_gem else None
            self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()
            self.logits = nn.Linear(self.nb_ft, num_classes)

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
            elif "coatnet" in self.encoder.name:
                conv = self.encoder.stem.conv1
            elif "efficientnet" in self.encoder.name:
                conv = self.encoder.conv_stem
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
            elif "coatnet" in self.encoder.name:
                self.encoder.stem.conv1 = new_conv
            elif "efficientnet" in self.encoder.name:
                self.encoder.conv_stem = new_conv

    def increase_stride(self):
        """
        Increase the stride of the first layer of the encoder.
        """
        if "nfnet" in self.encoder.name:
            self.encoder.model.stem_conv1.stride = (4, 4)
        elif "efficientnet" in self.encoder.name:
            self.encoder.model.conv_stem.stride = (4, 4)
        elif "resnet" in self.encoder.name or "resnext" in self.encoder.name:
            try:
                self.encoder.model.conv1[0].stride = (4, 4)
            except Exception:
                self.encoder.model.conv1.stride = (4, 4)
        elif "convnext" in self.encoder.name:
            print("VERIFY")
            self.encoder.stem[0].stride = (2, 2)
            self.encoder.stem[0].padding = (4, 4)
        elif "maxvit" in self.encoder.name or "maxxvit" in self.encoder.name:
            self.encoder.model.stem.conv1.stride = (4, 4)
        elif "coatnet" in self.encoder.name:
            self.encoder.model.stem.conv1.stride = (4, 4)
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
        if self.head == "freq_att":
            return self.freq_att(fts)

        if self.use_gem:
            fts = self.global_pool(fts)[:, :, 0, 0]
        else:  # Avg pooling
            while len(fts.size()) > 2:
                fts = fts.mean(-1)

        fts = self.dropout(fts)
        return self.logits(fts)

    def forward(self, x):
        """
        Forward function for the model.

        Args:
            x (torch.Tensor): Input waves of shape [batch_size x max_len].

        Returns:
            torch.Tensor: Logits for the primary classes of shape [batch_size x num_classes].
        """
        melspec = self.ft_extractor(x).unsqueeze(1)
        fts = self.encoder(melspec)
        logits = self.get_logits(fts)
        return logits
