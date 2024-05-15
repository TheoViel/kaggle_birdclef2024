import os
import h5py
import torch
import librosa
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


from params import CLASSES


class WaveDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
        secondary_labels_weight=0.0,
        normalize="std",
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
                if (
                    self.self_mixup and len(wave) > self.max_len * 2
                ):  # Mix start with end
                    start_1 = (
                        self.sample_start_end(wave, at_start=True)
                        if self.random_crop
                        else 0
                    )
                    start_2 = (
                        self.sample_start_end(wave, at_start=False)
                        if self.random_crop
                        else len(wave) - self.max_len
                    )
                    # start_2 = np.random.randint(start_1 + self.max_len, len(wave) - self.max_len)
                    wave = (
                        wave[start_1: start_1 + self.max_len]
                        + wave[start_2: start_2 + self.max_len]
                    ) / 2
                else:  # Start or end
                    start = (
                        self.sample_start_end(wave, at_start=None)
                        if self.random_crop
                        else 0
                    )
                    wave = wave[start: start + self.max_len]
            return np.array(wave)

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
                        if self.random_crop
                        else 0
                    )
                    start_2 = np.random.randint(
                        start_1 + self.max_len, len(wave) - self.max_len
                    )
                    wave = (
                        wave[start_1: start_1 + self.max_len]
                        + wave[start_2: start_2 + self.max_len]
                    ) / 2
                else:
                    start = (
                        np.random.randint(0, len(wave) - self.max_len)
                        if self.random_crop
                        else 0
                    )
                    wave = wave[start: start + self.max_len]
            return np.array(wave)

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
                    y[CLASSES.index(label)] = (
                        1 if use_secondary else self.secondary_labels_weight
                    )
                    y_aux[CLASSES.index(label)] = 1
                except ValueError:  # Not in considered classes
                    pass
        return y, y_aux

    def get_targets(self):
        targets = [self._get_target(i)[0].numpy() for i in range(self.__len__())]
        targets_sec = [
            self._get_target(i, use_secondary=True)[0].numpy()
            for i in range(self.__len__())
        ]
        return np.array(targets), np.array(targets_sec)

    def __getitem__(self, idx):
        wave = self._get_wave(idx)

        if self.normalize == "librosa":
            wave = librosa.util.normalize(wave)

        y, y_aux = self._get_target(idx)

        if self.transforms is not None:
            wave = self.transforms(wave)

        wave = torch.from_numpy(wave)
        w = self.sample_weights[idx]

        if self.normalize == "std":
            wave = wave / max(torch.std(wave), 1e-6)

        return wave, y, y_aux, w


class PLDataset(Dataset):
    def __init__(
        self,
        files,
        transforms=None,
        normalize="std",
        max_len=32000 * 5,
        folder="../input/unlabeled_features/unlabeled_soundscapes/",
    ):
        super().__init__()

        self.transforms = transforms
        self.folder = folder

        # Parameters
        self.normalize = normalize
        self.max_len = max_len

        self.load_files(files)

    def load_files(self, files):
        dfs = []
        self.targets = None
        for f in files:
            df = pd.read_csv(f + ".csv")
            dfs.append(df[["row_id"]])
            if len(df.columns) == 183:
                tgt = df[CLASSES].values
            else:
                tgt = np.load(f + ".npy")

            if self.targets is None:
                self.targets = tgt
            else:
                self.targets += tgt
        self.targets /= len(files)

        assert all(
            [(df["row_id"].values == dfs[0]["row_id"].values).all() for df in dfs]
        ), "PLs are not indexed identically"
        self.ids = dfs[0]["row_id"].values

    def __len__(self):
        return len(self.ids)

    def _get_wave(self, audio, end):
        with h5py.File(self.folder + audio, "r") as f:
            assert os.path.exists(self.folder + audio)
            wave = f["au"]
            wave = wave[end - self.max_len: end]

            if len(wave) < self.max_len:  # Pad
                pad_len = self.max_len - len(wave)
                wave = np.pad(np.array(wave), (0, pad_len)) if pad_len else wave

            return np.array(wave)

    def __getitem__(self, idx):
        audio, end = self.ids[idx].split("_")

        y = self.targets[idx]

        y = torch.from_numpy(y)
        y_aux = torch.zeros(len(CLASSES)).float()

        wave = self._get_wave(audio + ".hdf5", int(end) * 32000)
        if self.normalize == "librosa":
            wave = librosa.util.normalize(wave)

        if self.transforms is not None:
            wave = self.transforms(wave)
        wave = torch.from_numpy(wave)

        if self.normalize == "std":
            wave = wave / max(torch.std(wave), 1e-6)

        return wave, y, y_aux, 1
