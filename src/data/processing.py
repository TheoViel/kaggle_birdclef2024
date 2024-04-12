import os
import h5py
import librosa

from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed

try:
    import noisereduce as nr
except ImportError:
    print("`noisereduce` was not imported")


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def create_target_path(target_root, source_path):
    splitted_source_path = source_path.split("/")
    filename = os.path.splitext(splitted_source_path[-1])[0]
    target_path = os.path.join(target_root, splitted_source_path[-2], filename + ".hdf5")
    return target_path


def get_load_librosa_save_h5py(do_normalize, **kwargs):
    def load_librosa_save_h5py(load_path, save_path):
        if os.path.exists(save_path):
            return

        au, sr = librosa.load(load_path, **kwargs)
        if do_normalize:
            au = librosa.util.normalize(au)
        with h5py.File(save_path, "w") as data_file:
            data_file.create_dataset("au", data=au)
            # data_file.create_dataset("sr", data=sr)
            # data_file.create_dataset("do_normalize", data=int(do_normalize))
        # except Exception as e:
        #         print(f"Failed to load {load_path} with {e}")

    return load_librosa_save_h5py


def get_librosa_load(
    do_normalize,
    do_noisereduce=False,
    pos_dtype=None,
    return_au_len=False,
    **kwargs,
):
    def librosa_load(path):
        # assert kwargs["sr"] == 32_000
        try:
            au, sr = librosa.load(path, **kwargs)
            if do_noisereduce:
                try:
                    au = nr.reduce_noise(y=deepcopy(au), sr=sr)
                    if do_normalize:
                        au = librosa.util.normalize(au)
                    return au, sr
                except Exception as e:
                    print(f"{e} was catched while `reduce_noise`")
                    au, sr = librosa.load(path, **kwargs)
            if do_normalize:
                au = librosa.util.normalize(au)
            if pos_dtype is not None:
                au = au.astype(pos_dtype)
            if return_au_len:
                au = len(au)
            return au, sr
        except Exception as e:
            print(f"librosa_load failed with {e}")
            return None, None

    return librosa_load


def load_pp_audio(
    name,
    sr=None,
    normalize=True,
    do_noisereduce=False,
    pos_dtype=None,
    res_type="kaiser_best",
    validate_sr=None,
):
    # assert sr == 32_000
    au, sr = librosa.load(name, sr=sr)
    if validate_sr is not None:
        assert sr == validate_sr
    if do_noisereduce:
        try:
            au = nr.reduce_noise(y=deepcopy(au), sr=sr, res_type=res_type)
            if normalize:
                au = librosa.util.normalize(au)
            return au
        except Exception as e:
            print(f"{e} was catched while `reduce_noise`")
            au, sr = librosa.load(name, sr=sr)
    if normalize:
        au = librosa.util.normalize(au)
    if pos_dtype is not None:
        au = au.astype(pos_dtype)
    return au


def parallel_librosa_load(
    audio_pathes,
    n_cores=32,
    return_sr=True,
    return_audio=True,
    do_normalize=False,
    use_tqdm=True,
    **kwargs,
):
    assert return_sr or return_audio
    complete_out = ProgressParallel(n_jobs=n_cores, total=len(audio_pathes), use_tqdm=use_tqdm)(
        delayed(get_librosa_load(do_normalize=do_normalize, **kwargs))(el_path)
        for el_path in audio_pathes
    )
    if return_sr and return_audio:
        return complete_out
    elif return_audio:
        return [el[0] for el in complete_out]
    elif return_sr:
        return [el[1] for el in complete_out]
