import h5py
import torch
import librosa
import numpy as np

from torch.utils.data import Dataset


from params import CLASSES


class WaveDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
        secondary_labels_weight=0.,
        normalize=True,
        max_len=32000,
        self_mixup=False,
        random_crop=False,
        train=False,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.transforms = transforms

        self.labels = df["primary_label"].values
        self.secondary_labels = df["secondary_labels"].values
        self.paths = df["path_ft"].values
        self.sample_weights = (df["rating"].values + 1) / 6

        # Parameters
        self.secondary_labels_weight = secondary_labels_weight
        self.normalize = normalize
        self.max_len = max_len
        self.self_mixup = self_mixup
        self.random_crop = random_crop
        self.train = train

        if not self.train:
            assert not self.random_crop, "Ensure random_crop is disabled for val"
            assert not self.self_mixup, "Ensure self_mixup is disabled for val"

    def __len__(self):
        return len(self.df)

    def sample_start_end(self, wave, at_start=None, shift=32000):
        at_start = np.random.random() < 0.5 if at_start is None else at_start
        if at_start:
            start = np.random.randint(0, min(shift, len(wave) - self.max_len))
        else:
            start = np.random.randint(
                max(0, len(wave) - self.max_len - shift), len(wave) - self.max_len
            )
        return start

    def _get_wave(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            wave = f["au"]

            if len(wave) <= self.max_len:  # Pad
                # pad_len = (
                #     np.random.randint(0, self.max_len - len(wave) + 1) if self.train
                #     else self.max_len - len(wave)
                # )
                # wave = np.pad(np.array(wave), (self.max_len - len(wave) - pad_len, pad_len))

                pad_len = self.max_len - len(wave)
                wave = np.pad(np.array(wave), (0, pad_len)) if pad_len else wave

            else:  # Random start-end crop
                if self.self_mixup and len(wave) > self.max_len * 2:  # Mix start with end
                    start_1 = self.sample_start_end(wave, at_start=True) if self.random_crop else 0
                    start_2 = (
                        self.sample_start_end(wave, at_start=False) if self.random_crop
                        else len(wave) - self.max_len
                    )
                    # start_2 = np.random.randint(start_1 + self.max_len, len(wave) - self.max_len)
                    wave = (
                        wave[start_1: start_1 + self.max_len] +
                        wave[start_2: start_2 + self.max_len]
                    ) / 2
                else:  # Start or end
                    start = (
                        self.sample_start_end(wave, at_start=None)
                        if self.random_crop else 0
                    )
                    wave = wave[start: start + self.max_len]

            if self.normalize:
                wave = librosa.util.normalize(wave)

            return wave

    def _get_wave_old(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            wave = f["au"]

            if len(wave) <= self.max_len:  # Pad
                pad_len = self.max_len - len(wave)
                wave = np.pad(np.array(wave), (0, pad_len)) if pad_len else wave

            else:  # Random crop
                if self.self_mixup and len(wave) > self.max_len * 2:
                    start_1 = (
                        np.random.randint(0, len(wave) // 2 - self.max_len)
                        if self.random_crop else 0
                    )
                    start_2 = np.random.randint(start_1 + self.max_len, len(wave) - self.max_len)
                    wave = (
                        wave[start_1: start_1 + self.max_len] +
                        wave[start_2: start_2 + self.max_len]
                    ) / 2
                else:
                    start = (
                        np.random.randint(0, len(wave) - self.max_len)
                        if self.random_crop else 0
                    )
                    wave = wave[start: start + self.max_len]

            if self.normalize:
                wave = librosa.util.normalize(wave)

            return wave

    def _get_target(self, idx, use_secondary=False):
        y = torch.zeros(len(CLASSES)).float()
        y_aux = torch.zeros(len(CLASSES)).float()

        labels = self.labels[idx]
        if not isinstance(labels, list):
            labels = [labels]
        for label in labels:
            if label not in ["nocall", "test_soundscapes"]:
                y[CLASSES.index(label)] = 1

        if self.secondary_labels_weight or use_secondary:
            for label in self.secondary_labels[idx]:
                try:
                    y[CLASSES.index(label)] = 1 if use_secondary else self.secondary_labels_weight
                    y_aux[CLASSES.index(label)] = 1
                except ValueError:  # Not in considered classes
                    pass
        return y, y_aux

    def get_targets(self):
        targets = [self._get_target(i)[0].numpy() for i in range(self.__len__())]
        targets_sec = [
            self._get_target(i, use_secondary=True)[0].numpy() for i in range(self.__len__())
        ]
        return np.array(targets), np.array(targets_sec)

    def __getitem__(self, idx):
        wave = self._get_wave(idx)
        # wave_no_aug = torch.from_numpy(wave.copy())

        if self.normalize:
            wave = librosa.util.normalize(wave)

        y, y_aux = self._get_target(idx)

        if self.transforms is not None:
            wave = self.transforms(wave)

        wave = torch.from_numpy(wave)
        w = self.sample_weights[idx]
        return wave, y, y_aux, w  # wave_no_aug


class WaveInfDataset(Dataset):
    def __init__(
        self,
        df,
        normalize=True,
        max_len=32000,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.paths = df["path"].values
        self.slices = df["slice"].values
        self.normalize = normalize
        self.max_len = max_len

        self.waves = {}

    def __len__(self):
        return len(self.df)

    def _get_wave(self, idx):
        try:
            return self.waves[self.paths[idx]]
        except KeyError:
            wave, sr = librosa.load(self.paths[idx], sr=32000)

            if len(self.waves) > 100:
                self.waves = {}  # clear memory

            self.waves[self.paths[idx]] = wave
        return wave

    def __getitem__(self, idx):
        wave = self._get_wave(idx)

        wave = wave[self.slices[idx][0]: self.slices[idx][1]]

        if len(wave) <= self.max_len:  # Pad
            pad_len = self.max_len - len(wave)
            wave = np.pad(np.array(wave), (0, pad_len)) if pad_len else wave

        if self.normalize:
            wave = librosa.util.normalize(wave)
            # wave = (wave - wave.mean()) / (wave.std() + 1e-6)

        wave = torch.from_numpy(wave)
        return wave, 1, 1
