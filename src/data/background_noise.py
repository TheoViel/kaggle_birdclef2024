# Adapted from:
# https://github.com/VSydorskyy/BirdCLEF_2023_1st_place/blob/main/code_base/augmentations/transforms.py

import math
import librosa
import numpy as np
import pandas as pd

from glob import glob
from os.path import join as pjoin

# from data.transforms import AudioTransform
from data.processing import parallel_librosa_load

SAMPLE_RATE = 32000


ESC50_CATS_TO_INCLUDE = [
    "dog",
    "rain",
    "insects",
    "hen",
    "engine",
    "hand_saw",
    "pig",
    "rooster",
    "sea_waves",
    "cat",
    "crackling_fire",
    "thunderstorm",
    "chainsaw",
    "train",
    "sheep",
    "wind",
    "footsteps",
    "frog",
    "cow",
    "crickets",
]


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class BackgroundNoise(AudioTransform):
    def __init__(
        self,
        background_regex=None,
        esc50_root=None,
        esc50_df_path=None,
        esc50_cats_to_include=ESC50_CATS_TO_INCLUDE,
        always_apply=False,
        p=0.5,
        min_level=0.25,
        max_level=0.75,
        sr=32000,
        normalize=True,
        verbose=False,
        glob_recursive=False,
    ):
        super().__init__(always_apply, p)
        assert min_level < max_level
        assert 0 < min_level < 1
        assert 0 < max_level < 1
        if background_regex is None and (
            esc50_root is None or esc50_df_path is None
        ):
            raise ValueError(
                "background_regex OR esc50_root AND esc50_df_path should be defined"
            )
        if background_regex is not None:
            sample_names = glob(background_regex, recursive=glob_recursive)
        else:
            sample_df = pd.read_csv(esc50_df_path)
            if esc50_cats_to_include is not None:
                sample_df = sample_df[
                    sample_df.category.isin(esc50_cats_to_include)
                ]
            sample_names = [
                pjoin(esc50_root, el) for el in sample_df.filename.tolist()
            ]
        self.samples = parallel_librosa_load(
            sample_names,
            return_sr=False,
            sr=sr,
            do_normalize=True,
        )
        self.normalize = normalize
        self.min_max_levels = (min_level, max_level)
        self.verbose = verbose

    @staticmethod
    def crop_sample(sample, crop_shape):
        start = np.random.randint(0, sample.shape[0] - crop_shape)
        return sample[start: start + crop_shape]

    def apply(self, y: np.ndarray, **params):
        # TODO: It is a dirty hack BUT by some reasons we have corrupted samples
        # In order not to use it make sure that all samples are Okey
        back_sample = None
        while back_sample is None:
            back_sample = self.samples[np.random.randint(len(self.samples))]

        if y.shape[0] < back_sample.shape[0]:
            back_sample = self.crop_sample(back_sample, y.shape[0])
        elif y.shape[0] > back_sample.shape[0]:
            repeat_times = math.ceil(y.shape[0] / back_sample.shape[0])
            back_sample = np.concatenate([back_sample] * repeat_times)
            if back_sample.shape[0] > y.shape[0]:
                back_sample = self.crop_sample(back_sample, y.shape[0])

        back_amp = np.random.uniform(*self.min_max_levels)
        if self.verbose:
            print(f"BackgroundNoise. back_amp: {back_amp}")
        augmented = y * (1 - back_amp) + back_sample * back_amp

        if self.normalize:
            augmented = librosa.util.normalize(augmented)
        return augmented
