import os
import time
import torch
import warnings
import argparse
import pandas as pd

from data.preparation import prepare_data, prepare_xenocanto_data
from util.torch import init_distributed
from util.logger import create_logger, save_config, prepare_log_folder, init_neptune

from params import DATA_PATH, LOG_PATH


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        help="learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.,
        help="Weight decay",
    )
    return parser.parse_args()


class Config:
    """
    Parameters used for training
    """

    # General
    seed = 42
    verbose = 1
    device = "cuda"
    save_weights = True

    # Data
    duration = 5
    aug_strength = 1

    melspec_config = {
        "sample_rate": 32000,
        "n_mels": 128,
        "f_min": 20,
        "n_fft": 2048,
        "hop_length": 512,
        "normalized": True,
    }

    mixup_config = {
        "p": 0,
        "additive": True,
        "alpha": 4
    }

    spec_augment_config = {
        "freq_mask": {
            "mask_max_length": 10,
            "mask_max_masks": 3,
            "p": 0.3,
        },
        "time_mask": {
            "mask_max_length": 20,
            "mask_max_masks": 3,
            "p": 0.3,
        },
    }

    # k-fold
    k = 4
    folds_file = f"../input/folds_{k}.csv"
    selected_folds = [0, 1, 2, 3]

    # Model
    name = "tf_efficientnetv2_s"  # convnextv2_tiny maxvit_tiny_tf_384
    pretrained_weights = None

    num_classes = 282
    drop_rate = 0.1
    drop_path_rate = 0.1
    n_channels = 1
    head = "gem"

    # Training
    loss_config = {
        "name": "bce",
        "weighted": False,
        "smoothing": 0.0,
        "activation": "sigmoid",
    }

    data_config = {
        "batch_size": 32,
        "val_bs": 32,
        "mix": "cutmix",
        "num_classes": num_classes,
        "num_workers": 8,
    }

    optimizer_config = {
        "name": "Ranger",
        "lr": 5e-4,
        "warmup_prop": 0.0,
        "betas": (0.9, 0.999),
        "max_grad_norm": 0.1,
        "weight_decay": 0.0,
    }

    epochs = 40

    use_fp16 = True
    verbose = 1
    verbose_eval = 50 if data_config["batch_size"] >= 16 else 100

    fullfit = False
    n_fullfit = 1


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    config = Config
    init_distributed(config)

    if config.local_rank == 0:
        print("\nStarting !")
    args = parse_args()

    if not config.distributed:
        device = 0
        time.sleep(device)
        print("Using GPU ", device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        assert torch.cuda.device_count() == 1

    log_folder = None
    if config.local_rank == 0:
        log_folder = prepare_log_folder(LOG_PATH)
        print(f'\n -> Logging results to {log_folder}\n')

    if args.model:
        config.name = args.model
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.optimizer_config["lr"] = args.lr

    df = prepare_data(DATA_PATH)
    df_xc = prepare_xenocanto_data(DATA_PATH)
    df = pd.concat([df, df_xc], ignore_index=True)

    run = None
    if config.local_rank == 0:
        run = init_neptune(config, log_folder)

        create_logger(directory=log_folder, name="logs.txt")

        save_config(config, log_folder + "config.json")
        if run is not None:
            run["global/config"].upload(log_folder + "config.json")

        print("Device :", torch.cuda.get_device_name(0), "\n")

        print(f"- Model {config.name}")
        print(f"- Epochs {config.epochs}")
        print(
            f"- Learning rate {config.optimizer_config['lr']:.1e}   (n_gpus={config.world_size})"
        )
        print("\n -> Training\n")

    from training.main import k_fold

    df = df.head(1000)
    k_fold(config, df, log_folder=log_folder, run=run)

    if config.local_rank == 0:
        print("\nDone !")
