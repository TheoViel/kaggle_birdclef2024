import gc
import time
import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup

from data.loader import define_loaders
from training.losses import BirdLoss
from training.optim import define_optimizer
from util.torch import sync_across_gpus
from util.metrics import macro_auc


def evaluate(
    model,
    val_loader,
    loss_config,
    loss_fct,
    use_fp16=False,
    distributed=False,
    world_size=0,
    local_rank=0,
):
    """
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation set.
        loss_config (dict): Configuration parameters for the loss function.
        loss_fct (nn.Module): The loss function to compute the evaluation loss.
        use_fp16 (bool, optional): Whether to use mixed precision training. Defaults to False.
        distributed (bool, optional): Whether to use distributed training. Defaults to False.
        world_size (int, optional): Number of processes in distributed training. Defaults to 0.
        local_rank (int, optional): Local process rank in distributed training. Defaults to 0.

    Returns:
        preds (torch.Tensor): Predictions.
        val_loss (float): Validation loss.
    """
    model.eval()
    val_losses = []
    preds = []

    with torch.no_grad():
        for x, y, y_aux, w in val_loader:
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred = model(x.cuda())[0]
                loss = loss_fct(
                    y_pred.detach(), y.cuda(), secondary_mask=y_aux.cuda(), w=w.cuda()
                )

            val_losses.append(loss.detach())

            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)
            preds.append(y_pred.detach())

    val_losses = torch.stack(val_losses)
    preds = torch.cat(preds, 0)

    if distributed:
        val_losses = sync_across_gpus(val_losses, world_size)
        preds = sync_across_gpus(preds, world_size)
        torch.distributed.barrier()

    if local_rank == 0:
        preds = preds.cpu().numpy()
        val_loss = val_losses.cpu().numpy().mean()
        return preds, val_loss
    else:
        return 0, 0


def fit(
    model,
    train_dataset,
    val_dataset,
    data_config,
    loss_config,
    optimizer_config,
    pl_dataset=None,
    pl_config=None,
    epochs=1,
    verbose_eval=1,
    use_fp16=False,
    distributed=False,
    local_rank=0,
    world_size=1,
    log_folder=None,
    run=None,
    fold=0,
):
    """
    Train the model.

    Args:
        model (nn.Module): The main model to train.
        train_dataset (Dataset): Dataset for training.
        val_dataset (Dataset): Dataset for validation.
        data_config (dict): Configuration parameters for data loading.
        loss_config (dict): Configuration parameters for the loss function.
        optimizer_config (dict): Configuration parameters for the optimizer.
        pl_dataset (Dataset, optional): Dataset for pseudo-labels. Defaults to None.
        pl_config (dict): Configuration parameters for pseudo-labels. Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 1.
        verbose_eval (int, optional): Number of steps for verbose evaluation. Defaults to 1.
        use_fp16 (bool, optional): Whether to use mixed precision training. Defaults to False.
        distributed (bool, optional): Whether to use distributed training. Defaults to False.
        local_rank (int, optional): Local process rank in distributed training. Defaults to 0.
        world_size (int, optional): Number of processes in distributed training. Defaults to 1.
        log_folder (str, optional): Folder path for saving model weights. Defaults to None.
        run (neptune.Run, optional): Neptune run object for logging. Defaults to None.
        fold (int, optional): Fold number for tracking progress. Defaults to 0.

    Returns:
        preds (np array): Predictions.
        scores (dict): Scores.
    """
    scaler = torch.cuda.amp.GradScaler()

    optimizer = define_optimizer(
        model,
        optimizer_config["name"],
        lr=optimizer_config["lr"],
        lr_encoder=optimizer_config["lr"],  # optimizer_config["lr_encoder"],
        betas=optimizer_config["betas"],
        weight_decay=optimizer_config["weight_decay"],
    )

    train_loader, val_loader = define_loaders(
        train_dataset,
        val_dataset,
        batch_size=data_config["batch_size"],
        val_bs=data_config["val_bs"],
        num_workers=data_config["num_workers"],
        distributed=distributed,
        world_size=world_size,
        local_rank=local_rank,
    )

    # PL Mix
    if pl_dataset is not None:
        pl_loader = torch.utils.data.DataLoader(
            pl_dataset,
            batch_size=pl_config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        pl_iterator = iter(pl_loader)

    # LR Scheduler
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(optimizer_config["warmup_prop"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    loss_fct = BirdLoss(loss_config)

    auc, auc_s = 0, 0
    step, step_ = 1, 1
    avg_losses = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        if distributed:
            try:
                train_loader.sampler.set_epoch(epoch)
            except AttributeError:
                train_loader.batch_sampler.sampler.set_epoch(epoch)

        for x, y, y_aux, w in train_loader:
            if pl_dataset is not None:
                try:
                    x_pl, y_pl, y_aux_pl, w_pl = next(pl_iterator)
                except StopIteration:
                    pl_iterator = iter(pl_loader)
                    x_pl, y_pl, y_aux_pl, w_pl = next(pl_iterator)
                x = torch.cat([x, x_pl], 0)
                y = torch.cat([y, y_pl], 0)
                y_aux = torch.cat([y_aux, y_aux_pl], 0)
                w = torch.cat([w, w_pl], 0)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, y, y_aux, w = model(
                    x.cuda(),
                    y.cuda(),
                    y_aux.cuda(),
                    w.cuda(),
                )
                loss = loss_fct(y_pred, y, secondary_mask=y_aux, w=w)

            scaler.scale(loss).backward()
            avg_losses.append(loss.detach())

            scaler.unscale_(optimizer)

            if optimizer_config["max_grad_norm"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), optimizer_config["max_grad_norm"]
                )

            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()

            model.zero_grad(set_to_none=True)

            if distributed:
                torch.cuda.synchronize()

            if scale == scaler.get_scale():
                scheduler.step()

            step += 1
            if (step % verbose_eval) == 0 or step - 1 >= epochs * len(train_loader):
                if 0 <= epochs * len(train_loader) - step < verbose_eval:
                    continue

                avg_losses = torch.stack(avg_losses)
                if distributed:
                    avg_losses = sync_across_gpus(avg_losses, world_size)
                avg_loss = avg_losses.cpu().numpy().mean()

                preds, avg_val_loss = evaluate(
                    model,
                    val_loader,
                    loss_config,
                    loss_fct,
                    use_fp16=use_fp16,
                    distributed=distributed,
                    world_size=world_size,
                    local_rank=local_rank,
                )

                if local_rank == 0:
                    dt = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    step_ = step * world_size

                    preds = preds[: len(val_dataset)]
                    truth, truth_s = val_dataset.get_targets()
                    auc = macro_auc(truth, preds)
                    auc_s = macro_auc(truth_s, preds)

                    s = f"Epoch {epoch:02d}/{epochs:02d} (step {step_:04d}) \t"
                    s = s + f"lr={lr:.1e} \t t={dt:.0f}s \t loss={avg_loss:.3f}"
                    s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                    s = s + f"    auc={auc:.3f}" if auc else s
                    s = s + f"    auc_s={auc_s:.3f}" if auc_s else s
                    print(s)

                if run is not None:
                    run[f"fold_{fold}/train/epoch"].log(epoch, step=step_)
                    run[f"fold_{fold}/train/loss"].log(avg_loss, step=step_)
                    run[f"fold_{fold}/train/lr"].log(lr, step=step_)
                    if not np.isnan(avg_val_loss):
                        run[f"fold_{fold}/val/loss"].log(avg_val_loss, step=step_)
                    run[f"fold_{fold}/val/auc"].log(auc, step=step_)
                    run[f"fold_{fold}/val/auc_s"].log(auc_s, step=step_)

                start_time = time.time()
                avg_losses = []
                model.train()

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    if distributed:
        torch.distributed.barrier()

    return preds, {"auc": auc, "auc_s": auc_s}
