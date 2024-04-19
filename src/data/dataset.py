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
        self.train = train

    def __len__(self):
        return len(self.df)

    def _get_wave(self, idx):
        with h5py.File(self.paths[idx], "r") as f:
            wave = f["au"]

            if len(wave) <= self.max_len:  # Pad
                pad_len = self.max_len - len(wave)
                wave = np.pad(np.array(wave), (0, pad_len)) if pad_len else wave
            else:  # Random crop
                if self.self_mixup and len(wave) > self.max_len * 2:
                    start_1 = np.random.randint(0, len(wave) // 2 - self.max_len)
                    start_2 = np.random.randint(start_1 + self.max_len, len(wave) - self.max_len)

                    # print(start_1, start_2)

                    wave = (
                        wave[start_1: start_1 + self.max_len] +
                        wave[start_2: start_2 + self.max_len]
                    ) / 2
                else:
                    start = np.random.randint(0, len(wave) - self.max_len) if self.train else 0
                    wave = wave[start: start + self.max_len]

            if self.normalize:
                wave = librosa.util.normalize(wave)

            return wave

    def _get_target(self, idx, use_secondary=False):
        y = torch.zeros(len(CLASSES)).float()

        label = self.labels[idx]
        if label not in ["nocall", "test_soundscapes"]:
            y[CLASSES.index(self.labels[idx])] = 1

        if self.secondary_labels_weight or use_secondary:
            for label in self.secondary_labels[idx]:
                try:
                    y[CLASSES.index(label)] = 1 if use_secondary else self.secondary_labels_weight
                except ValueError:  # Not in considered classes
                    pass
        return y

    def get_targets(self):
        targets = [self._get_target(i).numpy() for i in range(self.__len__())]
        targets_sec = [
            self._get_target(i, use_secondary=True).numpy() for i in range(self.__len__())
        ]
        return np.array(targets), np.array(targets_sec)

    def __getitem__(self, idx):
        wave = self._get_wave(idx)
        # wave_no_aug = torch.from_numpy(wave.copy())

        if self.normalize:
            wave = librosa.util.normalize(wave)

        y = self._get_target(idx)

        if self.transforms is not None:
            wave = self.transforms(wave)

        wave = torch.from_numpy(wave)
        w = self.sample_weights[idx]
        return wave, y, w  # wave_no_aug


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

        wave = torch.from_numpy(wave)
        return wave, 1, 1
