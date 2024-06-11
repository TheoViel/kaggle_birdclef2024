import librosa
import numpy as np
import librosa.display as lid
import IPython.display as ipd
import matplotlib.pyplot as plt

from fastcluster import linkage
from scipy.spatial.distance import squareform


def load_audio(filepath, sr=32000):
    """
    Load audio file.

    Args:
        filepath (str): Path to the audio file.
        sr (int, optional): Target sampling rate. Defaults to 32000.

    Returns:
        numpy.ndarray: Loaded audio data.
    """
    audio, orig_sr = librosa.load(filepath, sr=None)

    if sr != orig_sr:
        audio = librosa.resample(audio, orig_sr, sr)

    audio = audio.astype("float32").ravel()
    return audio


def display_audio(audio, sr=32000, title="", duration=10):
    """
    Display audio waveform and play audio.

    Args:
        audio (str or numpy.ndarray): Path to the audio file or audio data.
        sr (int, optional): Sampling rate. Defaults to 32000.
        title (str, optional): Title for the audio plot. Defaults to "".
        duration (int, optional): Duration of audio to display in seconds. Defaults to 10.

    Returns:
        IPython.display.Audio: Audio widget.
    """
    if isinstance(audio, str):
        title = title if len(title) else audio
        audio = load_audio(audio, sr=sr)

    if duration:
        audio = audio[: sr * duration]

    plt.figure(figsize=(12, 3))
    plt.title(title)
    lid.waveshow(audio, sr=sr)
    plt.xlabel("")
    plt.show()

    return ipd.Audio(audio, rate=sr)


def plot_spectrogram(melspec, params, show_colorbar=False):
    """
    Plot mel spectrogram.

    Args:
        melspec (numpy.ndarray): Mel spectrogram.
        params (dict): Parameters for plotting.
        show_colorbar (bool, optional): Whether to show colorbar. Defaults to False.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    img = librosa.display.specshow(
        melspec,
        sr=params.get("sr", 32000),
        hop_length=params.get("hop_length", 512),
        x_axis="time",
        y_axis="linear",
        fmin=params.get('f_min', 20),
        fmax=params.get('f_max', 20000),
        ax=ax,
    )

    if show_colorbar:
        fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


def seriation(Z, N, cur_index):
    """
    Compute seriation.

    Args:
        Z (numpy.ndarray): Linkage matrix.
        N (int): Number of samples.
        cur_index (int): Current index.

    Returns:
        list: List of indices.
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_mat, method="ward"):
    """
    Compute seriated distance matrix.

    Args:
        dist_mat (numpy.ndarray): Distance matrix.
        method (str, optional): Linkage method. Defaults to "ward".

    Returns:
        tuple: Seriated distance matrix, result order, and result linkage.
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def plot_corr(correlations, model_names, reorder=True, res_order=None, figsize=(15, 15), clip=1.):
    """
    Plot correlation matrix.

    Args:
        correlations (numpy.ndarray): Correlation matrix.
        model_names (list): List of model names.
        reorder (bool, optional): Whether to reorder models based on correlations. Defaults to True.
        res_order (list, optional): Result order. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to (15, 15).
        clip (float, optional): Value to clip correlation matrix. Defaults to 1.

    Returns:
        list: Result order.
    """
    if reorder:
        m = 1.0 - 0.5 * (correlations + correlations.T)
        m[np.diag_indices_from(m)] = 0.0
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(m, "complete")

        names_order = [model_names[res_order[i]] for i in range(len(model_names))]
        corr = correlations[res_order, :][:, res_order]
    else:
        corr = correlations
        names_order = model_names

    if res_order is not None:
        names_order = [model_names[res_order[i]] for i in range(len(model_names))]
        corr = correlations[res_order, :][:, res_order]

    plt.figure(figsize=figsize)

    corr = np.clip(corr, 0, clip)
    plt.imshow(corr)
    plt.xticks([i for i in range(len(model_names))], names_order, rotation=-75)
    plt.yticks([i for i in range(len(model_names))], names_order)

    for i in range(len(model_names)):
        for j in range(len(model_names)):
            c = corr[j, i]
            col = "white" if c < (corr.min() + clip) / 2 else "black"
            if c:
                plt.text(i, j, f"{1 if c == clip else c:.3f}", va="center", ha="center", c=col)

    plt.show()
    return res_order
