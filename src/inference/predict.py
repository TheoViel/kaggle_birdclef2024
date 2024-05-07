import torch
import librosa
import numpy as np

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader


def predict(
    model,
    dataset,
    loss_config,
    batch_size=64,
    device="cuda",
    use_fp16=False,
    num_workers=8,
):
    """
    Perform inference using a single model and generate predictions for the given dataset.

    Args:
        model (torch.nn.Module): Trained model for inference.
        dataset (torch.utils.data.Dataset): Dataset for which to generate predictions.
        loss_config (dict): Configuration for loss function and activation.
        batch_size (int, optional): Batch size for prediction. Defaults to 64.
        device (str, optional): Device for inference, 'cuda' or 'cpu'. Defaults to 'cuda'.
        use_fp16 (bool, optional): Whether to use mixed-precision inference. Defaults to False.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 8.

    Returns:
        np array [N x C]: Predicted probabilities for each class for each sample.
        list: Empty list, placeholder for the auxiliary task.
    """
    model.eval()
    preds = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for img, _, _ in tqdm(loader):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred = model(img.to(device))
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

            # Get probabilities
            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)

            preds.append(y_pred.detach().cpu().numpy())
    return np.concatenate(preds)


def infer_onnx(ort_session, x, output_names=["output"], input_name="x"):
    x = ort_session.run(output_names, {input_name: x.numpy()})[0]
    return x


def load_sample(path, evaluate=False, sr=32000, duration=5, normalize=True):
    wave, sr = librosa.load(path, sr=sr)

    if evaluate:
        if len(wave) > sr * duration:
            wave = wave[: sr * duration][None]
        else:
            wave = np.pad(wave, (0, sr * duration - len(wave)))[None]
    else:
        wave = wave.reshape(-1, sr * duration)

    if normalize:
        wave = np.array([librosa.util.normalize(w) for w in wave])
        # wave = np.array([(w - w.mean()) / (w.std() + 1e-6)  for w in wave])

    wave = torch.from_numpy(wave)
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
                y_pred = model.get_logits(torch.from_numpy(fts))
            elif runtime == "onnx":
                fts = infer_onnx(session, x)
                y_pred = model.get_logits(torch.from_numpy(fts))
            else:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    fts = model.encoder(x)
                    y_pred = model.get_logits(fts)
            y_pred = y_pred.detach().cpu().numpy()
            preds.append(y_pred)

    return np.mean(preds, 0)
