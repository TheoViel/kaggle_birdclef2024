import gc
import torch
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from training.train import fit
from model_zoo.models import define_model

from data.preparation import add_xeno_low_freq, upsample_low_freq
from data.dataset import WaveDataset, PLDataset
from data.transforms import get_transfos

from util.torch import seed_everything, count_parameters, save_model_weights


def train(config, df_train, df_val, fold, log_folder=None, run=None):
    """
    Train a classification model.

    Args:
        config (Config): Configuration parameters for training.
        df_train (pandas DataFrame): Metadata for training dataset.
        df_val (pandas DataFrame): Metadata for validation dataset.
        fold (int): Fold number for cross-validation.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.

    Returns:
        tuple: A tuple containing predictions and metrics.
    """
    transforms = get_transfos(strength=config.aug_strength)

    train_dataset = WaveDataset(
        df_train,
        transforms=transforms,
        secondary_labels_weight=config.secondary_labels_weight,
        normalize=config.normalize,
        max_len=config.melspec_config["sample_rate"] * config.train_duration,
        self_mixup=config.self_mixup,
        random_crop=config.random_crop,
        train=True,
    )

    val_dataset = WaveDataset(
        df_val,
        normalize=config.normalize,
        max_len=config.melspec_config["sample_rate"] * config.duration,
        train=False,
    )

    pl_dataset = None
    if config.use_pl:
        pls = []
        for f in config.pl_config["folders"]:
            if f.endswith('.csv'):
                pls.append(pd.read_csv(f))
            elif "fullfit" not in str(fold):
                pls.append(pd.read_csv(f + f"pl_sub_{fold}.csv"))
            else:
                pls += [pd.read_csv(f + f"pl_sub_fullfit_{i}.csv")for i in range(5)]
        pl_dataset = PLDataset(pls, normalize=config.normalize)

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(
            ".pt"
        ) or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = config.pretrained_weights + f"{config.name}_{fold}.pt"
    else:
        pretrained_weights = None

    model = define_model(
        config.name,
        config.melspec_config,
        head=config.head,
        aug_config=config.aug_config,
        num_classes=config.num_classes,
        n_channels=config.n_channels,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        pretrained_weights=pretrained_weights,
        reduce_stride=config.reduce_stride,
        norm=config.norm,
        top_db=config.top_db,
        exportable=config.exportable,
        verbose=(config.local_rank == 0),
    ).cuda()

    if config.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    model.zero_grad(set_to_none=True)
    model.train()

    n_parameters = count_parameters(model)
    if config.local_rank == 0:
        print(f"    -> {len(train_dataset)} training birdcalls")
        print(f"    -> {len(val_dataset)} validation birdcalls")
        print(f"    -> {n_parameters} trainable parameters\n")

    preds, metrics = fit(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        pl_dataset=pl_dataset,
        pl_config=config.pl_config,
        epochs=config.epochs,
        verbose_eval=config.verbose_eval,
        use_fp16=config.use_fp16,
        distributed=config.distributed,
        local_rank=config.local_rank,
        world_size=config.world_size,
        log_folder=log_folder,
        run=run,
        fold=fold,
    )

    if (log_folder is not None) and (config.local_rank == 0):
        save_model_weights(
            model.module if config.distributed else model,
            f"{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()

    return preds, metrics


def k_fold(config, df, log_folder=None, run=None):
    """
    Perform k-fold cross-validation training for a classification model.

    Args:
        config (dict): Configuration parameters for training.
        df (pandas DataFrame): Main dataset metadata.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.
    """
    if "fold" not in df.columns:
        folds = pd.read_csv(config.folds_file)
        df = df.merge(folds, how="left")

    all_metrics = []
    for fold in range(config.k):
        if fold in config.selected_folds:
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n"
                )
            seed_everything(config.seed + fold)

            df_train = df[df["fold"] != fold].reset_index(drop=True)
            df_val = df[df["fold"] == fold].reset_index(drop=True)

            if config.upsample_low_freq_xc:
                extra = add_xeno_low_freq(df[df["fold"] != fold])
                df_train = pd.concat([df_train, extra], ignore_index=True)
            elif config.upsample_low_freq:
                extra = upsample_low_freq(df[df["fold"] != fold])
                df_train = pd.concat([df_train, extra], ignore_index=True)

            if len(df) <= 1000:
                df_train, df_val = df, df

            preds, metrics = train(
                config,
                df_train,
                df_val,
                fold,
                log_folder=log_folder,
                run=run,
            )
            all_metrics.append(metrics)

            if config.local_rank == 0:
                if log_folder is None:
                    return
                np.save(log_folder + f"pred_val_{fold}", preds)
                df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)

    if config.local_rank == 0 and len(config.selected_folds):
        print("\n-------------   CV Scores  -------------\n")

        for k in all_metrics[0].keys():
            avg = np.mean([m[k] for m in all_metrics])
            print(f"- {k} score\t: {avg:.3f}")
            if run is not None:
                run[f"global/{k}"] = avg

        if run is not None:
            run["global/logs"].upload(log_folder + "logs.txt")

        np.save(log_folder + f"pred_val_{fold}", preds)
        df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)

    if config.fullfit and len(config.selected_folds) in [0, 4]:
        for ff in range(config.n_fullfit):
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                )
            seed_everything(config.seed + ff)

            train(
                config,
                df,
                df.head(200).reset_index(drop=True),
                f"fullfit_{ff}",
                log_folder=log_folder,
                run=run,
            )

    if run is not None:
        print()
        run.stop()
