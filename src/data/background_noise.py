# Adapted from:
# https://github.com/VSydorskyy/BirdCLEF_2023_1st_place/blob/main/code_base/augmentations/transforms.py

import os
import math
import glob
import librosa
import numpy as np
import pandas as pd

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
        root=None,
        cats_to_include=ESC50_CATS_TO_INCLUDE,
        always_apply=False,
        p=0.5,
        min_level=0.25,
        max_level=0.75,
        sr=32000,
        normalize=True,
        verbose=False,
        n_samples=1000,
    ):
        super().__init__(always_apply, p)
        assert min_level < max_level
        assert 0 < min_level < 1
        assert 0 < max_level < 1

        df_path = glob.glob(os.path.join(root, "*.csv"))[0]
        df = pd.read_csv(df_path)

        if cats_to_include is not None and "category" in df.columns:
            df = df[df["category"].isin(cats_to_include)]

        samples = [os.path.join(root, "audio", el) for el in df["filename"].tolist()]
        if len(samples) > n_samples:
            samples = np.random.choice(samples, n_samples)
        self.samples = parallel_librosa_load(
            samples,
            return_sr=False,
            sr=sr,
            do_normalize=True,
            use_tqdm=False,
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
