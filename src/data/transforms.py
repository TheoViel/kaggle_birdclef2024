import os
import numpy as np
from audiomentations import Gain, GainTransition

from data.noise_augs import (
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
)
from data.background_noise import BackgroundNoise
from params import DATA_PATH


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        y = trns(y)
        return y


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


def noise_transfos(p=1.0):
    return OneOf(
        [
            NoiseInjection(max_noise_level=0.04, p=p),
            GaussianNoise(min_snr=5, max_snr=20, p=p),
            PinkNoise(min_snr=5, max_snr=20, p=p),
            AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=p),
            AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=p),
        ]
    )


def gain_transfos(p=1.0):
    return OneOf(
        [
            lambda x: Gain(min_gain_in_db=-15, max_gain_in_db=15, p=p)(
                x, sample_rate=32000
            ),
            lambda x: GainTransition(min_gain_in_db=-15, max_gain_in_db=15, p=p)(
                x, sample_rate=32000
            ),
        ],
    )


def background_transfos(folders=None, p=1.0):
    if folders is None:
        # paths = [p for p in paths if os.path.exists(p)]
        # if strength > 2 or not len(paths):
        paths = [
            # DATA_PATH + "esc50/",
            DATA_PATH + "nocall_2023/audio/",
            DATA_PATH + "background_noise/birdclef2021_nocall/",
            DATA_PATH + "background_noise/birdclef2020_nocall/",
            DATA_PATH + "background_noise/freefield/",
            DATA_PATH + "background_noise/warblrb/",
            DATA_PATH + "background_noise/birdvox/",
            DATA_PATH + "background_noise/rainforest/",
            DATA_PATH + "background_noise/environment/",
        ]
    else:
        paths = [DATA_PATH + f for f in folders]

    paths = [p for p in paths if os.path.exists(p)]

    return OneOf(
        [
            BackgroundNoise(
                root=path,
                normalize=True,
                p=p,
                n_samples=300 // len(paths),
            )
            for path in paths
        ]
    )


def get_transfos(augment=True, normalize=True, strength=1):
    """
    Returns transformations.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        strength (int, optional): Augmentation strength level. Defaults to 1.

    Returns:
        albumentation transforms: Transforms.
    """
    if not augment or strength <= 0:
        return None

    if strength == 1:
        transfos = Compose(
            [
                # background_transfos(strength=1, folders=["unlabeled_soundscapes/"], p=1),
                background_transfos(p=0.5),
            ]
        )
    elif strength == 2:
        transfos = Compose(
            [
                background_transfos(p=0.5),
                gain_transfos(p=0.3),
                noise_transfos(p=0.2),
            ]
        )
    elif strength == 3:
        transfos = Compose(
            [
                background_transfos(p=0.75),
                gain_transfos(p=0.75),
                noise_transfos(p=0.5),
            ]
        )
    else:
        raise NotImplementedError

    return transfos
