import glob
import numpy as np

from audiomentations import Gain, GainTransition, AddBackgroundNoise

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


def background_transfos(p=1.0):
    birdclef2021_nocall = glob.glob("../input/background_noise/birdclef2021_nocall/*")
    birdclef2020_nocall = glob.glob("../input/background_noise/birdclef2020_nocall/*")
    freefield = glob.glob("../input/background_noise/freefield/*")
    warblrb = glob.glob("../input/background_noise/warblrb/*")
    birdvox = glob.glob("../input/background_noise/birdvox/*")
    rainforest = glob.glob("../input/background_noise/rainforest/*")
    environment = glob.glob("../input/background_noise/environment/*")

    return OneOf(
        [
            BackgroundNoise(
                root=DATA_PATH + "nocall_2023/",
                normalize=True,
                p=p,
            ),
            BackgroundNoise(
                root=DATA_PATH + "esc50/",
                normalize=True,
                p=p,
            ),
            lambda x: AddBackgroundNoise(
                birdclef2021_nocall + birdclef2020_nocall,
                min_snr_in_db=0,
                max_snr_in_db=3,
                p=p,
            )(x, sample_rate=32000),
            lambda x: AddBackgroundNoise(
                freefield + warblrb + birdvox,
                min_snr_in_db=0,
                max_snr_in_db=3,
                p=p,
            )(x, sample_rate=32000),
            lambda x: AddBackgroundNoise(
                rainforest + environment,
                min_snr_in_db=0,
                max_snr_in_db=3,
                p=p
            )(x, sample_rate=32000),
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
        transfos = background_transfos(p=0.5)
    elif strength == 2:
        transfos = Compose(
            [
                background_transfos(p=0.5),
                gain_transfos(p=0.5),
                noise_transfos(p=0.25),
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
