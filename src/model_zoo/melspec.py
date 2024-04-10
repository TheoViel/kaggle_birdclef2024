import math
import torch
import numpy as np
import torch.nn as nn

from time import time
from typing import Optional
from torchaudio.transforms import (
    MelScale,
    AmplitudeToDB,
    MelSpectrogram,
    FrequencyMasking,
    TimeMasking,
)

try:
    from nnAudio.features import CQT1992v2
    from nnAudio.features.stft import STFT as nnAudioSTFT
    from nnAudio.features.stft import STFTBase
    from nnAudio.utils import create_fourier_kernels
except ImportError:
    print("`nnAudio` was not imported")


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        params,
        top_db=80,
        exportable=False,
        quantizable=False,
        spec_extractor="melspec",
        spec_augment_config=None,
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

        if spec_extractor == "melspec":
            if exportable:
                spectrogram = TraceableMelspec
                params[quantizable] = quantizable
            else:
                spectrogram = MelSpectrogram
        elif spec_extractor == "cqt":
            spectrogram = CQT1992v2
        else:
            raise NotImplementedError(f"{spec_extractor} not implemented")

        amplitude_to_db = QuantizableAmplitudeToDB if quantizable else AmplitudeToDB

        self.extractor = nn.Sequential(
            spectrogram(**params),
            amplitude_to_db(top_db=top_db),
            NormalizeMelSpec(exportable=exportable),
        )

        if spec_augment_config is not None:
            self.freq_mask = CustomFreqMasking(**spec_augment_config["freq_mask"])
            self.time_mask = CustomTimeMasking(**spec_augment_config["time_mask"])
        else:
            self.time_mask = nn.Identity()
            self.freq_mask = nn.Identity()

    def forward(self, x):
        melspec = self.extractor(x)

        self.freq_mask(melspec)
        self.time_mask(melspec)

        return melspec


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6, exportable=False):
        super().__init__()
        self.eps = eps
        self.exportable = exportable

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        if self.exportable:
            norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
            return (Xstd - norm_min) / (norm_max - norm_min + self.eps)
        else:
            norm_min, norm_max = (
                Xstd.min(-1)[0].min(-1)[0],
                Xstd.max(-1)[0].max(-1)[0],
            )
            fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
                (norm_max - norm_min)
            )
            V = torch.zeros_like(Xstd)
            if fix_ind.sum():
                V_fix = Xstd[fix_ind]
                norm_max_fix = norm_max[fix_ind, None, None]
                norm_min_fix = norm_min[fix_ind, None, None]
                V_fix = torch.max(
                    torch.min(V_fix, norm_max_fix),
                    norm_min_fix,
                )
                V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
                V[fix_ind] = V_fix
            return V


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
        quantizable: bool = False,
    ):
        super().__init__()
        if quantizable:
            self.spectrogram = QuantizableSTFT(
                n_fft=n_fft,
                win_length=win_length,
                freq_bins=None,
                hop_length=hop_length,
                window="hann",
                freq_scale="no",
                # Do not define `fmin` and `fmax`, because freq_scale = "no"
                center=center,
                pad_mode=pad_mode,
                sr=sample_rate,
                trainable=trainable,
                output_format="Complex",
                verbose=True,
            )
        else:
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
                verbose=True,
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


class QuantizableSTFT(STFTBase):
    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        freq_bins=None,
        hop_length=None,
        window="hann",
        freq_scale="no",
        center=True,
        pad_mode="reflect",
        fmin=50,
        fmax=6000,
        sr=22050,
        trainable=False,
        output_format="Complex",
        verbose=True,
    ):

        super().__init__()

        # Trying to make the default setting same as librosa
        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = int(win_length // 4)

        self.output_format = output_format
        self.trainable = trainable
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length
        self.trainable = trainable
        start = time()

        # Create filter windows for stft
        (
            kernel_sin,
            kernel_cos,
            self.bins2freq,
            self.bin_list,
            window_mask,
        ) = create_fourier_kernels(
            n_fft,
            win_length=win_length,
            freq_bins=freq_bins,
            window=window,
            freq_scale=freq_scale,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            verbose=verbose,
        )

        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)

        # Applying window functions to the Fourier kernels
        window_mask = torch.tensor(window_mask)
        wsin = kernel_sin * window_mask
        wcos = kernel_cos * window_mask

        self.spec_imag_conv = nn.Conv1d(
            wsin.shape[1],
            wsin.shape[0],
            wsin.shape[2],
            stride=self.stride,
            bias=False,
        )
        self.spec_real_conv = nn.Conv1d(
            wcos.shape[1],
            wcos.shape[0],
            wcos.shape[2],
            stride=self.stride,
            bias=False,
        )
        # Set the weights and bias manually
        with torch.no_grad():
            self.spec_imag_conv.weight.copy_(wsin)
            self.spec_real_conv.weight.copy_(wcos)

        if not self.trainable:
            for param in self.spec_imag_conv.parameters():
                param.requires_grad = False
            for param in self.spec_real_conv.parameters():
                param.requires_grad = False

        if verbose:
            print(
                "STFT kernels created, time used = {:.4f} seconds".format(
                    time() - start
                )
            )
        else:
            pass

        if self.center:
            if self.pad_mode == "constant":
                self.padding_node = nn.ConstantPad1d(self.pad_amount, 0)
            elif self.pad_mode == "reflect":
                self.padding_node = nn.ReflectionPad1d(self.pad_amount)

    def forward(self, x):
        """
        Convert a batch of waveforms to spectrograms.

        Parameters
        ----------
        x : torch tensor
            Input signal should be in either of the following shapes.\n
            1. ``(len_audio)``\n
            2. ``(num_audio, len_audio)``\n
            3. ``(num_audio, 1, len_audio)``
            It will be automatically broadcast to the right shape
        """
        self.num_samples = x.shape[-1]

        x = x[:, None, :]
        if self.center:
            x = self.padding_node(x)
        spec_imag = self.spec_imag_conv(x)
        spec_real = self.spec_real_conv(x)

        # remove redundant parts
        spec_real = spec_real[:, : self.freq_bins, :]
        spec_imag = spec_imag[:, : self.freq_bins, :]

        if self.output_format == "Magnitude":
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable:
                return torch.sqrt(
                    spec + 1e-8
                )  # prevent Nan gradient when sqrt(0) due to output=0
            else:
                return torch.sqrt(spec)

        elif self.output_format == "Complex":
            return torch.stack(
                (spec_real, -spec_imag), -1
            )  # Remember the minus sign for imaginary part

        elif self.output_format == "Phase":
            return torch.atan2(
                -spec_imag + 0.0, spec_real
            )  # +0.0 removes -0.0 elements, which leads to error in calculating phase


class QuantizableAmplitudeToDB(torch.nn.Module):
    r"""Turn a tensor from the power/amplitude scale to the decibel scale.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        stype (str, optional): scale of input tensor (``"power"`` or ``"magnitude"``). The
            power being the elementwise square of the magnitude. (Default: ``"power"``)
        top_db (float or None, optional): minimum negative cut-off in decibels.  A reasonable
            number is 80. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        >>> waveform_db = transform(waveform)
    """

    __constants__ = ["multiplier", "amin", "ref_value", "db_multiplier"]

    def __init__(self, stype: str = "power", top_db: Optional[float] = None) -> None:
        super().__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Numerically stable implementation from Librosa.

        https://librosa.org/doc/latest/generated/librosa.amplitude_to_db.html

        Args:
            x (Tensor): Input tensor before being converted to decibel scale.

        Returns:
            Tensor: Output tensor in decibel scale.
        """
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        if self.top_db is not None:
            # Expand batch
            shape = x_db.size()
            x_db = x_db.reshape(-1, shape[-3], shape[-2], shape[-1])

            x_db = torch.max(
                x_db,
                (x_db.amax(dim=(-3, -2, -1)) - self.top_db).view(-1, 1, 1, 1),
            )

            # Repack batch
            x_db = x_db.reshape(shape)

        return x_db


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
