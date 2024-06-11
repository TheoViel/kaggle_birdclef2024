import torch
import librosa
import numpy as np


def infer_onnx(ort_session, x, output_names=["output"], input_name="x"):
    """
    Infers output from an ONNX session.

    Args:
        ort_session (onnxruntime.InferenceSession): ONNX inference session.
        x (numpy.ndarray or torch.Tensor): Input tensor.
        output_names (list, optional): Names of the output nodes. Defaults to ["output"].
        input_name (str, optional): Name of the input node. Defaults to "x".

    Returns:
        numpy.ndarray: Output tensor.
    """
    x = ort_session.run(output_names, {input_name: x.numpy()})[0]
    return x


def load_sample(path, evaluate=False, sr=32000, duration=5, normalize="librosa"):
    """
    Load audio sample from file.

    Args:
        path (str): Path to the audio file.
        evaluate (bool, optional): Whether to evaluate. Defaults to False.
        sr (int, optional): Sample rate. Defaults to 32000.
        duration (int, optional): Duration of the audio sample in seconds. Defaults to 5.
        normalize (str, optional): Normalization method. Defaults to "librosa".

    Returns:
        torch.Tensor: Loaded audio sample.
    """
    wave, sr = librosa.load(path, sr=sr)

    if evaluate:
        if len(wave) > sr * duration:
            wave = wave[: sr * duration][None]
        else:
            wave = np.pad(wave, (0, sr * duration - len(wave)))[None]
    else:
        try:
            wave = wave.reshape(-1, sr * duration)
        except Exception:
            wave = wave[: int(len(wave) // (sr * duration) * (sr * duration))]
            wave = wave.reshape(-1, sr * duration)

    if normalize == "librosa":
        wave = np.array([librosa.util.normalize(w) for w in wave])

    wave = torch.from_numpy(wave)

    if normalize == "std":
        wave = wave / torch.clamp(torch.std(wave, axis=1, keepdims=True), min=1e-6)
    return wave


def infer_sample(wave, models, sessions, device="cpu", use_fp16=False):
    """
    Infer predictions for a single audio sample using the provided models and sessions.

    Args:
        wave (str or torch.Tensor): Path to the audio file or pre-loaded audio tensor.
        models (list): List of tuples containing model instances and their runtimes.
        sessions (list): List of inference sessions corresponding to the models.
        device (str, optional): Device for inference. Defaults to "cpu".
        use_fp16 (bool, optional): Whether to use FP16 precision. Defaults to False.

    Returns:
        np.ndarray: Predictions for the input audio sample.
    """
    if isinstance(wave, str):
        wave = load_sample(wave)

    preds = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            melspec = models[0][0].ft_extractor(wave.to(device))[0].unsqueeze(1)

        for (model, runtime), session in zip(models, sessions):

            if model.n_channels == 3:
                pos = (
                    torch.linspace(0.0, 1.0, melspec.size(2))
                    .to(melspec.device)
                    .view(1, 1, -1, 1)
                )
                pos = pos.expand(melspec.size(0), 1, melspec.size(2), melspec.size(3))
                x = torch.cat([melspec, melspec, pos], 1)
            else:
                x = melspec

            if runtime == "openvino":
                fts = session.infer(inputs=[x.numpy()])["output"]
            elif runtime == "onnx":
                fts = infer_onnx(session, x)
            else:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    fts = model.encoder(x)

            if model.logits is not None:
                if isinstance(fts, np.ndarray):
                    fts = torch.from_numpy(fts)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    y_pred = model.get_logits(fts)
            else:
                y_pred = fts

            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            preds.append(y_pred)
    return np.array(preds)
