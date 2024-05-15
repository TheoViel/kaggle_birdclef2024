import torch
import librosa
import numpy as np


def infer_onnx(ort_session, x, output_names=["output"], input_name="x"):
    x = ort_session.run(output_names, {input_name: x.numpy()})[0]
    return x


def load_sample(path, evaluate=False, sr=32000, duration=5, normalize="librosa"):
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
        wave = wave / (torch.std(wave, axis=1, keepdims=True) + 1e-6)
    return wave


def infer_sample(wave, models, sessions, device="cpu", use_fp16=False):
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
