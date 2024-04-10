import librosa
import librosa.display as lid
import IPython.display as ipd
import matplotlib.pyplot as plt


def load_audio(filepath, sr=32000):
    audio, orig_sr = librosa.load(filepath, sr=None)

    if sr != orig_sr:
        audio = librosa.resample(audio, orig_sr, sr)

    audio = audio.astype("float32").ravel()
    return audio


def display_audio(audio, sr=32000, title="", duration=10):
    if isinstance(audio, str):
        title = title if len(title) else audio
        audio = load_audio(audio, sr=sr)

    audio = audio[: sr * duration]

    plt.figure(figsize=(12, 3))
    plt.title(title)
    lid.waveshow(audio, sr=sr)
    plt.xlabel("")
    plt.show()

    return ipd.Audio(audio, rate=sr)


def plot_spectrogram(melspec, params, show_colorbar=False):
    fig, ax = plt.subplots(figsize=(15, 5))
    img = librosa.display.specshow(
        melspec,
        sr=params.get("sr", 32000),
        hop_length=params.get("hop_length", 512),
        x_axis="time",
        y_axis="linear",
        fmin=20,
        ax=ax,
    )

    if show_colorbar:
        fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()
