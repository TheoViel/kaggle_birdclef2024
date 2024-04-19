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
    train_duration = 5  # 15
    duration = 5
    aug_strength = 1
    self_mixup = True
    normalize = True  # False ??

    melspec_config = {
        "sample_rate": 32000,
        "n_mels": 128,  # 128, 224
        "f_min": 50,
        "f_max": 15000,
        "n_fft": 2048,
        "hop_length": 512,  # 716, 512, 417
        "normalized": False,
    }

    aug_config = {
        "specaug_freq": {
            "mask_max_length": 10,
            "mask_max_masks": 3,
            "p": 0.5,
        },
        "specaug_time": {
            "mask_max_length": 20,
            "mask_max_masks": 3,
            "p": 0.5,
        },
        "mixup":
        {
            "p_audio": 0.75,
            "p_spec": 0.25,
            "additive": True,
            "alpha": 4,
            "num_classes": 182,
        }
    }

    # k-fold
    k = 4
    folds_file = f"../input/folds_{k}.csv"
    selected_folds = [0, 1, 2, 3]

    # Model
    name = "tf_efficientnetv2_s"  # tf_efficientnetv2_s maxvit_tiny_tf_384 eca_nfnet_l0
    pretrained_weights = None

    num_classes = 182
    drop_rate = 0.2
    drop_path_rate = 0.2
    n_channels = 1
    head = "gem"
    reduce_stride = "256" in name

    # Training
    loss_config = {
        "name": "bce",
        "weighted": True,
        "smoothing": 0.,
        "top_k": 0,
        "activation": "sigmoid",  # "softmax"
    }
    secondary_labels_weight = 1. if loss_config["name"] == "bce" else 0.5  # 0.5 for ce / 1. for bce

    data_config = {
        "batch_size": 64,
        "val_bs": 256,
        "num_classes": num_classes,
        "num_workers": 8,
    }

    optimizer_config = {
        "name": "Ranger",
        "lr": 5e-3,
        "warmup_prop": 0.0,
        "betas": (0.9, 0.999),
        "max_grad_norm": 0.1,
        "weight_decay": 0.,
    }

    epochs = 20

    use_fp16 = True
    verbose = 1
    verbose_eval = 200

    fullfit = True
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

    # df = df.head(1000)
    k_fold(config, df, log_folder=log_folder, run=run)

    if config.local_rank == 0:
        print("\nDone !")
