import torch
import numpy as np
import torch.nn as nn

from typing import Optional
from torchaudio.transforms import (
    MelScale,
    AmplitudeToDB,
    MelSpectrogram,
    FrequencyMasking,
    TimeMasking,
)
from data.mix import Mixup

try:
    from nnAudio.features.stft import STFT as nnAudioSTFT
    from nnAudio.features.stft import STFTBase
except ImportError:
    STFTBase = nn.Module
    print("`nnAudio` was not imported")


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        params,
        aug_config=None,
        top_db=80,
        exportable=False,
        spec_extractor="melspec",
    ):
        """
        params={
            "sample_rate": 32000,
            "n_mels": 128,
            "f_min": 20,
            "n_fft": 2048,
            "hop_length": 512,
            "normalized": True,
        },
        """
        super().__init__()

        spectrogram = TraceableMelspec if exportable else MelSpectrogram
        self.extractor = spectrogram(**params)
        self.amplitude_to_db = AmplitudeToDB(top_db=top_db)
        # self.normalizer = MeanStdNorm()
        self.normalizer = MinMaxNorm()

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
        super().__init__()
        self.eps = eps

    def forward(self, X):
        min_ = torch.amax(X, dim=(1, 2), keepdim=True)
        max_ = torch.amin(X, dim=(1, 2), keepdim=True)
        return (X - min_) / (max_ - min_ + self.eps)


class MeanStdNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.reshape(X.size(0), -1).std(1, keepdim=True).unsqueeze(-1)
        return (X - mean) / (std + self.eps)


class TraceableMelspec(nn.Module):
    def __init__(
        self,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        # Mel params
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_fft: int = 400,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        trainable: bool = False,
    ):
        super().__init__()
        self.spectrogram = nnAudioSTFT(
            n_fft=n_fft,
            win_length=win_length,
            freq_bins=None,
            hop_length=hop_length,
            window="hann",
            freq_scale="no",
            # Do not define `fmin` and `fmax`, because freq_scale = "no"
            center=center,
            pad_mode=pad_mode,
            iSTFT=False,
            sr=sample_rate,
            trainable=trainable,
            output_format="Complex",
            verbose=False,
        )
        self.normalized = normalized
        self.power = power
        self.register_buffer(
            "window",
            torch.hann_window(win_length if win_length is not None else n_fft),
        )
        self.trainable = trainable
        self.mel_scale = MelScale(
            n_mels, sample_rate, f_min, f_max, n_fft // 2 + 1, norm, mel_scale
        )

    def forward(self, x):
        spec_f = self.spectrogram(x)
        if self.normalized:
            spec_f /= self.window.pow(2.0).sum().sqrt()
        if self.power is not None:
            # prevent Nan gradient when sqrt(0) due to output=0
            # Taken from nnAudio.features.stft.STFT
            eps = 1e-8 if self.trainable else 0.0
            spec_f = torch.sqrt(
                spec_f[:, :, :, 0].pow(2) + spec_f[:, :, :, 1].pow(2) + eps
            )
            if self.power != 1.0:
                spec_f = spec_f.pow(self.power)
        mel_spec = self.mel_scale(spec_f)
        return mel_spec


class CustomMasking(nn.Module):
    """
    Adapted from:
    https://github.com/VSydorskyy/BirdCLEF_2023_1st_place/blob/main/code_base/augmentations/spec_augment.py

    Simplified to always be inplace.
    """

    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0):
        super().__init__()
        assert isinstance(mask_max_masks, int) and mask_max_masks > 0
        self.mask_max_masks = mask_max_masks
        self.mask_max_length = mask_max_length
        self.mask_module = None
        self.p = p

    def forward(self, x):
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                n_applies = np.random.randint(low=1, high=self.mask_max_masks + 1)
                for _ in range(n_applies):
                    x[i: i + 1] = self.mask_module(x[i: i + 1])


class CustomTimeMasking(CustomMasking):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True):
        super().__init__(
            mask_max_length=mask_max_length,
            mask_max_masks=mask_max_masks,
            p=p,
        )
        self.mask_module = TimeMasking(time_mask_param=mask_max_length)


class CustomFreqMasking(CustomMasking):
    def __init__(self, mask_max_length: int, mask_max_masks: int, p=1.0, inplace=True):
        super().__init__(
            mask_max_length=mask_max_length,
            mask_max_masks=mask_max_masks,
            p=p,
        )
        self.mask_module = FrequencyMasking(freq_mask_param=mask_max_length)
