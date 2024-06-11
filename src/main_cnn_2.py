import os
import time
import torch
import warnings
import argparse
import pandas as pd

from data.preparation import prepare_data_2, add_xeno_low_freq
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
        default=0.0,
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
    use_xc = True
    upsample_low_freq = False
    remove_low_rating = True

    train_duration = 5  # 15, 5
    duration = 5
    random_crop = True  # True
    sampling = "start"
    wav_norm = "std"

    use_pl = True
    pl_config = {
        "folders": [
            "../logs/2024-05-16/3/",  # efficientvit_b1
            "../logs/2024-05-16/5/",  # mobilenetv2
            "../logs/2024-05-16/7/",  # efficientnet_b0
            "../logs/2024-05-16/9/",  # efficientnetv2_b0
            "../logs/2024-06-06/1/",  # efficientvit_b0
            "../logs/2024-06-06/3/",  # efficientvit_m3
            "../logs/2024-06-07/3/",  # 5 efficientvit_m3
        ],
        "batch_size": 32,
        "agg": "avg_pp",
    }

    melspec_config = {
        "sample_rate": 32000,
        "n_mels": 224,
        "f_min": 90,
        "f_max": 14000,
        "n_fft": 1536,
        "hop_length": 717,
        "win_length": 1024,
        "mel_scale": "htk",
        "power": 2.0,
    }
    norm = "simple"
    top_db = None

    aug_config = {
        "specaug_freq": {
            "mask_max_length": 10,
            "mask_max_masks": 3,
            "p": 0.0,
        },
        "specaug_time": {
            "mask_max_length": 20,
            "mask_max_masks": 3,
            "p": 0.0,
        },
        "mixup": {
            "p_audio": 0.25,
            "p_spec": 0.0,
            "additive": True,
            "alpha": 4,
            "num_classes": 182,
            "norm": wav_norm,
        },
    }

    # k-fold
    k = 4
    folds_file = f"../input/folds_{k}.csv"
    selected_folds = [0, 1, 2, 3]

    # Model
    name = "mnasnet_100"
    pretrained_weights = None

    num_classes = 182
    drop_rate = 0.2
    drop_path_rate = 0.2
    n_channels = 3
    head = "gem"
    reduce_stride = False

    # Training
    loss_config = {
        "name": "bce",
        "weighted": False,
        "mask_secondary": True,
        "smoothing": 0.,
        "activation": "sigmoid",
    }
    secondary_labels_weight = 0.0 if loss_config["mask_secondary"] else 1.0

    data_config = {
        "batch_size": 32,
        "val_bs": 256,
        "num_classes": num_classes,
        "num_workers": 8,
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 1e-3,
        "warmup_prop": 0.0,
        "betas": (0.9, 0.999),
        "max_grad_norm": 0.0,
        "weight_decay": 0.01,
    }

    epochs = 80

    use_fp16 = True
    verbose = 1
    verbose_eval = 100 if epochs <= 20 else 200

    fullfit = True
    n_fullfit = 5


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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
        print(f"\n -> Logging results to {log_folder}\n")

    if args.model:
        config.name = args.model
        config.head = "" if "vit" in args.model else "gem"
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.optimizer_config["lr"] = args.lr

    df = prepare_data_2(DATA_PATH)
    if config.use_xc:
        df_xc = add_xeno_low_freq(
            df, low_freq=500, remove_low_rating=config.remove_low_rating
        )
        df_previous_comps = pd.read_csv(DATA_PATH + "df_extra_comp.csv")
        df = pd.concat([df, df_xc, df_previous_comps], ignore_index=True)

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
