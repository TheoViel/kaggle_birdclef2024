# Adapted from: https://github.com/VSydorskyy/BirdCLEF_2023_1st_place

import os
import h5py
import librosa

from tqdm import tqdm
from joblib import Parallel


class ProgressParallel(Parallel):
    """
    Parallel processing with progress tracking.

    Args:
        use_tqdm (bool, optional): Whether to use tqdm for progress tracking. Defaults to True.
        total (int, optional): Total number of tasks. Defaults to None.
    """
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Calls the parallel processing function with tqdm for progress tracking.

        Returns:
            Any: Results of the parallel processing.
        """
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        """
        Prints the progress of the parallel processing.
        """
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def create_target_path(target_root, source_path):
    """
    Creates the target path for a file based on the source path.
    To save the hdf5 file.

    Args:
        target_root (str): Root directory for the target path.
        source_path (str): Path to the source file.

    Returns:
        str: Target path for the file.
    """
    splitted_source_path = source_path.split("/")
    filename = os.path.splitext(splitted_source_path[-1])[0]
    target_path = os.path.join(target_root, splitted_source_path[-2], filename + ".hdf5")
    return target_path


def get_load_librosa_save_h5py(do_normalize, **kwargs):
    """
    Returns a function to load audio using librosa, normalize it if specified,
    and save it in HDF5 format.

    Args:
        do_normalize (bool): Whether to normalize the audio.
        **kwargs: Additional keyword arguments to pass to librosa.load.

    Returns:
        function: Function.
    """
    def load_librosa_save_h5py(load_path, save_path):
        """
        Load audio using librosa, normalize if specified, and save it in HDF5 format.

        Args:
            load_path (str): Path to the audio file to load.
            save_path (str): Path to save the HDF5 file.

        Returns:
            None
        """
        if os.path.exists(save_path):
            return

        au, sr = librosa.load(load_path, **kwargs)
        if do_normalize:
            au = librosa.util.normalize(au)
        with h5py.File(save_path, "w") as data_file:
            data_file.create_dataset("au", data=au)
    return load_librosa_save_h5py
