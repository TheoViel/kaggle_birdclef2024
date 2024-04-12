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
        use_secondary_labels=True,
        normalize=True,
        max_len=32000,
        mixup_config=None,
        train=False,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.transforms = transforms

        self.labels = df["primary_label"].values
        self.secondary_labels = df["secondary_labels"].values
        self.paths = df["path_ft"].values

        # Parameters
        self.use_secondary_labels = use_secondary_labels
        self.normalize = normalize
        self.mixup_config = mixup_config if mixup_config is not None else {}
        self.max_len = max_len
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
                start = np.random.randint(0, len(wave) - self.max_len) if self.train else 0
                wave = wave[start: start + self.max_len]

            if self.normalize:
                wave = librosa.util.normalize(wave)

            return wave

    def _get_target(self, idx):
        y = torch.zeros(len(CLASSES)).float()

        label = self.labels[idx]
        if label != "nocall":
            y[CLASSES.index(self.labels[idx])] = 1

        if self.use_secondary_labels:
            for label in self.secondary_labels[idx]:
                try:
                    y[CLASSES.index(label)] = 1
                except ValueError:  # Not in considered classes
                    pass
        return y

    def get_targets(self):
        targets = [self._get_target(i).numpy() for i in range(self.__len__())]
        return np.array(targets)

    def _mixup(self, wave, target):
        mixup_idx = np.random.randint(0, self.__len__())

        mixup_wave = self._get_wave(mixup_idx)
        mixup_target = self._get_target(mixup_idx)

        mix_weight = np.random.beta(
            self.mixup_config["alpha"], self.mixup_config["alpha"]
        )
        wave = mix_weight * mixup_wave + (1 - mix_weight) * wave

        if self.mixup_config.get("additive", True):
            target = mixup_target + target
        else:
            target = mix_weight * mixup_target + (1 - mix_weight) * target

        target = torch.clamp(target, min=0, max=1.0)

        return wave, target

    def __getitem__(self, idx):
        wave = self._get_wave(idx)

        if self.normalize:
            wave = librosa.util.normalize(wave)

        y = self._get_target(idx)

        if np.random.random() < self.mixup_config.get("p", 0):
            wave, y = self._mixup(wave, y)

        if self.transforms is not None:
            wave = self.transforms(wave)

        wave = torch.from_numpy(wave)
        return wave, y, 1  # 1 is a placeholder for sample weight
