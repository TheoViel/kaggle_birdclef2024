import os
import h5py
import torch
import librosa
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from scipy.special import expit, logit

from params import CLASSES


class WaveDataset(Dataset):
    def __init__(
        self,
        df,
        transforms=None,
        secondary_labels_weight=0.0,
        normalize="std",
        max_len=32000,
        random_crop=False,
        sampling="start",
        train=False,
    ):
        """
        Dataset class for handling waveform data.

        Args:
            df (pd.DataFrame): DataFrame containing metadata including paths to waveform files.
            transforms (Optional[Callable]): Augs to apply to the waveforms. Defaults to None.
            secondary_labels_weight (float): Weight for secondary labels. Defaults to 0.0.
            normalize (str): Waveforms normalization ("std" or "librosa"). Defaults to "std".
            max_len (int): Maximum length of waveforms. Defaults to 32000.
            random_crop (bool): Whether to add randomness to cropping. Defaults to False.
            sampling (str): Sampling strategy ("random", "start" "start_end"). Defaults to "start".
            train (bool): Whether the dataset is used for training. Defaults to False.
        """
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
        self.random_crop = random_crop
        self.sampling = sampling
        self.train = train

        if not self.train:
            assert not self.random_crop, "Ensure random_crop is disabled for val"

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def sample_start_end(self, wave, at_start=None, shift=32000):
        """
        Sample a starting point for waveform cropping.
        Will either use the start or end of the sample.

        Args:
            wave (np.ndarray): The waveform array.
            at_start (Optional[bool]): Whether to sample from the start. Defaults to None.
            shift (int): Maximum shift for the random starting point. Defaults to 32000.

        Returns:
            int: The starting index for cropping.
        """
        at_start = np.random.random() < 0.5 if at_start is None else at_start
        if at_start:
            start = np.random.randint(0, min(shift, len(wave) - self.max_len))
        else:
            start = np.random.randint(
                max(0, len(wave) - self.max_len - shift), len(wave) - self.max_len
            )
        return start

    def _get_wave_start_end(self, idx, start=False):
        """
        Retrieve and process a waveform segment from a file.

        Args:
            idx (int): Index of the waveform file.
            start (bool): Whether to sample from the start. Defaults to False.

        Returns:
            np.ndarray: Processed waveform segment.
        """
        with h5py.File(self.paths[idx], "r") as f:
            wave = f["au"]

            if len(wave) <= self.max_len:  # Pad
                pad_len = self.max_len - len(wave)
                wave = np.pad(np.array(wave), (0, pad_len)) if pad_len else wave

            else:  # Random start-end crop
                start = (
                    self.sample_start_end(wave, at_start=start)
                    if self.random_crop
                    else 0
                )
                wave = wave[start: start + self.max_len]
            return np.array(wave)

    def _get_wave_random(self, idx):
        """
        Retrieve and process a random waveform segment from a file.

        Args:
            idx (int): Index of the waveform file.

        Returns:
            np.ndarray: Processed waveform segment.
        """
        with h5py.File(self.paths[idx], "r") as f:
            wave = f["au"]

            if len(wave) <= self.max_len:  # Pad
                pad_len = self.max_len - len(wave)
                wave = np.pad(np.array(wave), (0, pad_len)) if pad_len else wave

            else:  # Random crop
                start = (
                    np.random.randint(0, len(wave) - self.max_len)
                    if self.random_crop
                    else 0
                )
                wave = wave[start: start + self.max_len]
            return np.array(wave)

    def _get_target(self, idx, use_secondary=False):
        """
        Retrieve the primary and secondary labels for a given index.

        Args:
            idx (int): Index of the sample.
            use_secondary (bool, optional): Whether to include secondary labels. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The primary and auxiliary label tensors.
        """
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
        """
        Get all primary and secondary targets for the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of primary and secondary targets.
        """
        targets = [self._get_target(i)[0].numpy() for i in range(self.__len__())]
        targets_sec = [
            self._get_target(i, use_secondary=True)[0].numpy()
            for i in range(self.__len__())
        ]
        return np.array(targets), np.array(targets_sec)

    def __getitem__(self, idx):
        """
        Retrieve and process a waveform, along with its labels and weight.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Waveform.
            torch.Tensor: Label.
            torch.Tensor: Secondary label.
            float: Sample weight.
        """
        if self.sampling == "random":
            wave = self._get_wave_random(idx)
        elif self.sampling == "start":
            wave = self._get_wave_start_end(idx, start=True)
        elif self.sampling == "start_end":
            wave = self._get_wave_start_end(idx, start=None)
        else:
            raise NotImplementedError

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
        folder="",
        agg="avg"
    ):
        """
        Dataset class for handling pseudo-labeled soundscape data.

        Args:
            files (List[str]): List of file paths to CSV or NPY files containing labels.
            transforms (Optional[Callable]): Waveform augs. Defaults to None.
            normalize (str): Waveforms normalization ("std" or "librosa"). Defaults to "std".
            max_len (int): Maximum length of waveforms. Defaults to 32000.
            folder (str): Folder containing the waveform files. Defaults to "".
            agg (str): Aggregation method for PLs ("avg", "max", or "pp"). Defaults to "avg".
        """
        super().__init__()

        self.transforms = transforms
        self.folder = folder

        # Parameters
        self.normalize = normalize
        self.max_len = max_len
        self.agg = agg

        self.load_files(files)

    def load_files(self, files):
        """
        Load and aggregate data from specified files.

        Args:
            files (List[str]): List of file paths to load.

        Raises:
            AssertionError: If the loaded data is not indexed identically.
        """
        dfs = []
        self.targets = None
        for f in files:
            df = pd.read_csv(f + ".csv")
            dfs.append(df[["row_id"]])
            if len(df.columns) == 183:
                tgt = df[CLASSES].values
            else:
                tgt = np.load(f + ".npy")

                df.loc[:, CLASSES] = tgt

            if self.targets is None:
                self.targets = tgt
            else:
                if "avg" in self.agg:
                    self.targets += tgt
                elif self.agg == "max":  # max
                    self.targets = np.max([self.targets, tgt], 0)
        if "avg" in self.agg:
            self.targets /= len(files)

        if "pp" in self.agg:
            # print("PP")
            df.loc[:, CLASSES] = logit(self.targets)
            df["group"] = df["row_id"].apply(lambda x: int(x.split('_')[0]))

            dfg_max = df[["group"] + CLASSES].groupby('group').max().reset_index()
            dfg_mean = df[["group"] + CLASSES].groupby('group').mean().reset_index()

            delta = dfg_mean[CLASSES].mean(1) - dfg_max[CLASSES].mean(1)
            for c in CLASSES:
                dfg_max[c] += delta

            df = df.merge(dfg_max, how="left", on="group", suffixes=('', '_delta'))
            for c in CLASSES:
                df[c] = expit((df[c] + df[c + "_delta"]) / 2)

            self.targets = df[CLASSES].values

        assert all(
            [(df["row_id"].values == dfs[0]["row_id"].values).all() for df in dfs]
        ), "PLs are not indexed identically"
        self.ids = dfs[0]["row_id"].values

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.ids)

    def _get_wave(self, audio, end):
        """
        Retrieve and process a waveform segment from a file.

        Args:
            audio (str): File name of the waveform file.
            end (int): End position for waveform cropping.

        Returns:
            np.ndarray: Processed waveform segment.

        Raises:
            AssertionError: If the file does not exist.
        """
        with h5py.File(self.folder + audio, "r") as f:
            assert os.path.exists(self.folder + audio)
            wave = f["au"]
            wave = wave[end - self.max_len: end]

            if len(wave) < self.max_len:  # Pad
                pad_len = self.max_len - len(wave)
                wave = np.pad(np.array(wave), (0, pad_len)) if pad_len else wave

            return np.array(wave)

    def __getitem__(self, idx):
        """
        Retrieve and process a waveform, along with its labels and weight.

        Args:
            idx (int): Index of the sample.

        Returns:
            torch.Tensor: Waveform.
            torch.Tensor: Label.
            torch.Tensor: Secondary label.
            float: Sample weight.
        """
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
