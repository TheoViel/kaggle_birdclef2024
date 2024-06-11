import torch
import numpy as np
import torch.nn as nn

from torchaudio.transforms import (
    AmplitudeToDB,
    MelSpectrogram,
    FrequencyMasking,
    TimeMasking,
)
from data.mix import Mixup


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        params,
        aug_config=None,
        top_db=None,
        norm="min_max",
    ):
        """
        Feature extraction module.

        Args:
            params (dict): Parameters for the spectrogram.
            aug_config (dict, optional): Configuration for data augmentation. Defaults to None.
            top_db (float, optional): Threshold for computing the amplitude to dB. Defaults to None.
            norm (str, optional): Normalization method. Defaults to "min_max".
        """
        super().__init__()

        self.extractor = MelSpectrogram(**params)
        self.amplitude_to_db = AmplitudeToDB(top_db=top_db)

        if norm == "mean_std":
            self.normalizer = MeanStdNorm()
        elif norm == "min_max":
            self.normalizer = MinMaxNorm()
        elif norm == "simple":
            self.normalizer = SimpleNorm()
        else:
            self.normalizer = nn.Identity()

        if aug_config is not None:
            self.freq_mask = CustomFreqMasking(**aug_config["specaug_freq"])
            self.time_mask = CustomTimeMasking(**aug_config["specaug_time"])
            self.mixup_audio = Mixup(
                p=aug_config["mixup"].get("p_audio", 0), **aug_config["mixup"]
            )
            self.mixup_spec = Mixup(
                p=aug_config["mixup"].get("p_spec", 0), **aug_config["mixup"]
            )
        else:
            self.time_mask = nn.Identity()
            self.freq_mask = nn.Identity()
            self.mixup_audio = lambda w, x, y, z: (
                nn.Identity()(w),
                nn.Identity()(x),
                nn.Identity()(y),
                nn.Identity()(z),
            )
            self.mixup_spec = lambda w, x, y, z: (
                nn.Identity()(w),
                nn.Identity()(x),
                nn.Identity()(y),
                nn.Identity()(z),
            )

    def forward(self, x, y=None, y_aux=None, w=None):
        """
        Forward pass of the feature extractor.

        Args:
            x (torch.Tensor): Input audio data.
            y (torch.Tensor, optional): Target labels. Defaults to None.
            y_aux (torch.Tensor, optional): Auxiliary labels. Defaults to None.
            w (torch.Tensor, optional): Auxiliary weights. Defaults to None.

        Returns:
            torch.Tensor: Extracted features.
            torch.Tensor: Target labels.
            torch.Tensor: Auxiliary labels.
            torch.Tensor: Auxiliary weights.
        """
        if self.training:
            x, y, y_aux, w = self.mixup_audio(x, y, y_aux, w)

        with torch.cuda.amp.autocast(enabled=False):
            melspec = self.extractor(x.float())
            melspec = self.amplitude_to_db(melspec)
            melspec = self.normalizer(melspec)

        if self.training:
            melspec, y, y_aux, w = self.mixup_spec(melspec, y, y_aux, w)

            self.freq_mask(melspec)
            self.time_mask(melspec)

        return melspec, y, y_aux, w


class MinMaxNorm(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Module for performing min-max normalization on input data.

        Args:
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, X):
        """
        Forward pass of the min-max normalization module.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Normalized data.
        """
        min_ = torch.amax(X, dim=(1, 2), keepdim=True)
        max_ = torch.amin(X, dim=(1, 2), keepdim=True)
        return (X - min_) / (max_ - min_ + self.eps)


class SimpleNorm(nn.Module):
    def __init__(self):
        """
        Module for performing simple normalization on input data.
        """
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the simple normalization module.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Normalized data.
        """
        return (x - 40) / 80


class MeanStdNorm(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Module for performing mean and standard deviation normalization on input data.

        Args:
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps

    def forward(self, X):
        """
        Forward pass of the mean and standard deviation normalization module.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Normalized data.
        """
        mean = X.mean((1, 2), keepdim=True)
        std = X.reshape(X.size(0), -1).std(1, keepdim=True).unsqueeze(-1)
        return (X - mean) / (std + self.eps)


class CustomMasking(nn.Module):
    """
    Adapted from:
    https://github.com/VSydorskyy/BirdCLEF_2023_1st_place/blob/main/code_base/augmentations/spec_augment.py

    Simplified to always be inplace.
    """

    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0):
        """
        Module for applying custom masking to input data.

        Args:
            mask_max_length (int): Maximum length of the mask.
            mask_max_masks (int): Maximum number of masks to apply.
            p (float): Probability of applying the masking. Defaults to 1.0.
        """
        super().__init__()
        assert isinstance(mask_max_masks, int) and mask_max_masks > 0
        self.mask_max_masks = mask_max_masks
        self.mask_max_length = mask_max_length
        self.mask_module = None
        self.p = p

    def forward(self, x):
        """
        Forward pass of the custom masking module.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            None.
        """
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                n_applies = np.random.randint(low=1, high=self.mask_max_masks + 1)
                for _ in range(n_applies):
                    x[i: i + 1] = self.mask_module(x[i: i + 1])


class CustomTimeMasking(CustomMasking):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True):
        """
        Module for applying custom time masking to input data.

        Args:
            mask_max_length (int): Maximum length of the mask.
            mask_max_masks (int): Maximum number of masks to apply.
            p (float): Probability of applying the masking. Defaults to 1.0.
            inplace (bool): Whether to apply masking inplace. Defaults to True.
        """
        super().__init__(
            mask_max_length=mask_max_length,
            mask_max_masks=mask_max_masks,
            p=p,
        )
        self.mask_module = TimeMasking(time_mask_param=mask_max_length)


class CustomFreqMasking(CustomMasking):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True):
        """
        Module for applying custom frequency masking to input data.

        Args:
            mask_max_length (int): Maximum length of the mask.
            mask_max_masks (int): Maximum number of masks to apply.
            p (float): Probability of applying the masking. Defaults to 1.0.
            inplace (bool): Whether to apply masking inplace. Defaults to True.
        """
        super().__init__(
            mask_max_length=mask_max_length,
            mask_max_masks=mask_max_masks,
            p=p,
        )
        self.mask_module = FrequencyMasking(freq_mask_param=mask_max_length)
