import numpy as np
# import albumentations as albu
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


def get_transfos(augment=True, normalize=True, strength=1):
    """
    Returns transformations.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        strength (int, optional): Augmentation strength level. Defaults to 1.

    Returns:
        albumentation transforms: Transforms.
    """

    if augment and strength > 0:
        if strength == 1:
            augs = OneOf(
                [
                    BackgroundNoise(
                        p=0.5,
                        root=DATA_PATH + "nocall_2023/",
                        normalize=True,
                    ),
                    BackgroundNoise(
                        p=0.5,
                        root=DATA_PATH + "esc50/",
                        normalize=True,
                    ),
                ]
            )
        elif strength == 2:
            augs = OneOf(
                [
                    BackgroundNoise(
                        p=0.75,
                        root=DATA_PATH + "nocall_2023/",
                        normalize=True,
                    ),
                    BackgroundNoise(
                        p=0.75,
                        root=DATA_PATH + "esc50/",
                        normalize=True,
                    ),
                ]
            )
        else:
            raise NotImplementedError
    else:
        augs = None

    return augs
