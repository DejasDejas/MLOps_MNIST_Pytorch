from pathlib import Path
import json

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from ignite.engine import (Events,
                           create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.metrics import Accuracy, Loss, ConfusionMatrix
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.utils import manual_seed
import numpy as np
import datetime

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        raise RuntimeError(
            "This module requires either tensorboardX or torch >= 1.2.0. "
            "You may install tensorboardX with command: \n pip install "
            "tensorboardX \n"
            "or upgrade PyTorch using your package manager of choice "
            "(pip or conda)."
        ) from e

import os
from tqdm import tqdm

from src.trainer.utils import (
    log_model_weights,
    log_data_stats,
    log_model_grads,
    plot_classes_preds,
    gpu_config,
    print_num_params,
    plot_cm
)
import config.log as log
from config.config import ROOT_DIR

logger = log.setup_custom_logger(__name__)

SEED = 42


def score_function(engine):
    val_loss = engine.state.metrics["nll"]
    return -val_loss


def trainer(model, dataset, args):
    # arguments parameters:
    epochs = args.epochs
    lr = args.lr
    log_interval = args.log_interval
    log_dir = args.log_dir + datetime.datetime.now().strftime("%Y_%m_%d--%Hh%Mmn%S")
    model_dir = args.model_dir
    checkpoint_every = args.checkpoint_every
    resume_from = args.resume_from
    crash_iteration = args.crash_iteration
    tb_graph_model = args.tb_graph_model
    early_stopping = args.early_stopping

    # data:
    class_names = np.arange(10)
    train_loader = dataset.train_loader
    test_loader = dataset.test_loader

    # training log init:
    log_dir = os.path.join(ROOT_DIR, log_dir)
    model_dir = os.path.join(ROOT_DIR, model_dir)
    writer = SummaryWriter(log_dir=log_dir)
    pbar = tqdm(initial=0,
                leave=False,
                total=len(train_loader),
                desc=f"Epoch 0 - loss: {0:.4f} - lr: {lr:.4f}",
                )

    # model and training parameters config:
    model, device = gpu_config(model)
    print_num_params(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion),
        "cm": ConfusionMatrix(len(class_names))
    }

    # inspect the model using TensorBoard:
    if tb_graph_model:
        images, labels = next(train_loader.__iter__())
        writer.add_graph(model, images)

    # Setup trainer and evaluator
    supervised_trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    if early_stopping:
        # early stopping
        es_handler = EarlyStopping(
            patience=10, score_function=score_function, trainer=supervised_trainer
        )

    # Apply learning rate scheduling
    @supervised_trainer.on(Events.EPOCH_COMPLETED)
    def lr_step():
        lr_scheduler.step()

    @supervised_trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        _lr = optimizer.param_groups[0]["lr"]
        pbar.desc = f"Epoch {engine.state.epoch} - loss: {engine.state.output:.4f} - lr: {_lr:.4f}"
        pbar.update(log_interval)
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        writer.add_scalar("lr", _lr, engine.state.iteration)

    if crash_iteration > 0:
        @supervised_trainer.on(Events.ITERATION_COMPLETED(once=crash_iteration))
        def _(engine):  # sourcery skip: raise-specific-error
            raise Exception(f"STOP at {engine.state.iteration}")

    if resume_from is not None:
        @supervised_trainer.on(Events.STARTED)
        def _(engine):
            pbar.n = engine.state.iteration % engine.state.epoch_length

    @supervised_trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        _metrics = evaluator.state.metrics
        avg_accuracy = _metrics["accuracy"]
        avg_nll = _metrics["nll"]
        tqdm.write(
            f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )
        writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)
        # ...log a Matplotlib Figure showing the model's predictions on a
        # random mini-batch
        writer.add_figure(
            "predictions vs. actual",
            plot_classes_preds(model, images, labels, class_names),
            global_step=engine.state.epoch,
        )

    # _____________________________________ Prediction on test dataloader __________________________________________
    # Compute and tensorboard_logs validation metrics
    @supervised_trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_loader)
        _metrics = evaluator.state.metrics
        avg_accuracy = _metrics["accuracy"]
        avg_nll = _metrics["nll"]
        cm = _metrics['cm']
        cm = cm.numpy()
        cm = cm.astype(int)
        if early_stopping:
            evaluator.add_event_handler(Events.COMPLETED, es_handler)
        tqdm.write(
            f"Test Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )
        pbar.n = pbar.last_print_n = 0
        writer.add_scalar("validation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)
        writer.add_figure(
            "Confusion Matrix on test set",
            plot_cm(class_names, cm),
            global_step=engine.state.epoch,
        )

    # _____________________________________ Save results __________________________________________
    def save_results():
        evaluator.run(test_loader)
        _metrics = evaluator.state.metrics
        avg_accuracy = _metrics["accuracy"]
        cm = _metrics['cm']
        cm = cm.numpy()
        cm = cm.astype(int)
        _ = plot_cm(class_names, cm, save=True)

        json_file = os.path.join(ROOT_DIR, "reports", 'metrics', 'metrics.json')
        with open(json_file, 'w') as outfile:
            json.dump({"accuracy": avg_accuracy}, outfile)
        return

    save_results()
    # _____________________________________ Fin Prediction on test dataloader ________________________________________

    # Setup object to checkpoint
    objects_to_checkpoint = {
        "trainer": supervised_trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(model_dir, require_empty=False),
        n_saved=None,
        global_step_transform=lambda *_: supervised_trainer.state.epoch,
    )
    supervised_trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=checkpoint_every), training_checkpoint
    )

    # Setup logger to print and dump into file: model weights, model grads and data stats
    # - first 3 iterations
    # - 4 iterations after checkpointing
    # This helps to compare resumed training with checkpoint training
    def log_event_filter(_e, event):
        if event in [1, 2, 3]:
            return True
        elif 0 <= (event % (checkpoint_every * _e.state.epoch_length)) < 5:
            return True
        return False

    fp = Path(log_dir) / ("run.log" if resume_from is None else "resume_run.log")
    fp = fp.as_posix()
    for h in [log_data_stats, log_model_weights, log_model_grads]:
        supervised_trainer.add_event_handler(
            Events.ITERATION_COMPLETED(event_filter=log_event_filter),
            h,
            model=model,
            fp=fp,
        )

    if resume_from is not None:
        tqdm.write(f"Resume from the checkpoint: {resume_from}")
        checkpoint = torch.load(os.path.join(ROOT_DIR, resume_from))
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)
    try:
        # Synchronize random states
        manual_seed(SEED)
        supervised_trainer.run(train_loader, max_epochs=epochs)
    except Exception as _e:
        import traceback

        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        logger.info(template.format(type(_e).__name__, _e.args))
        logger.info(traceback.format_exc(), _e)

    @supervised_trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time():
        tqdm.write(
            f"{supervised_trainer.last_event_name.name} took "
            f"{supervised_trainer.state.times[supervised_trainer.last_event_name.name]} seconds "
        )

    pbar.close()
    writer.close()

    return
