# pylint: disable=[invalid-name, disable=import-error, no-name-in-module, unused-variable]
# pylint: disable=broad-except, too-many-statements, too-many-locals
"""System module."""
import datetime
import os
import traceback
import sys

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
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from ignite.engine import (Events,
                           create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.metrics import Accuracy, Loss, ConfusionMatrix
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.utils import manual_seed, setup_logger
from ignite.contrib.handlers.neptune_logger import (
    global_step_from_engine as global_step_from_engine_neptune,
    GradsScalarHandler as GradsScalarHandler_neptune,
    NeptuneLogger,
    WeightsScalarHandler as WeightsScalarHandler_neptune,
    OptimizerParamsHandler as OptimizerParamsHandler_neptune,
)
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine as global_step_from_engine_tensorboard,
    GradsScalarHandler as GradsScalarHandler_tensorboard,
    TensorboardLogger,
    WeightsScalarHandler as WeightsScalarHandler_tensorboard,
    OptimizerParamsHandler as OptimizerParamsHandler_tensorboard,
)

import numpy as np
from tqdm import tqdm
from src.trainer.utils import (
    gpu_config,
    print_num_params,
    plot_cm
)
from config.log import setup_custom_logger
from config.config import ROOT_DIR

logger = setup_custom_logger(__name__)

SEED = 42


def score_function(engine):
    """
    Score function for ignite.
    """
    val_loss = engine.state.metrics["nll"]
    return -val_loss


def logger_config(s_trainer, args, optimizer, model, log_interval, train_evaluator,
                  validation_evaluator, log_dir):
    """
    Configure neptune logger.
    """
    assert not (args.tensorboard_logs and args.neptune_logs),\
        "Please specify either tensorboard or neptune logs, not both."
    if args.tensorboard_logs:
        config_logger = TensorboardLogger(log_dir=log_dir)
        GradsScalarHandler = GradsScalarHandler_tensorboard
        WeightsScalarHandler = WeightsScalarHandler_tensorboard
        OptimizerParamsHandler = OptimizerParamsHandler_tensorboard
        global_step_from_engine = global_step_from_engine_tensorboard
        logger.info("Using TensorboardLogger")
    elif args.neptune_logs:
        config_logger = NeptuneLogger(
            api_token=args.neptune_api_token,
            project_name="dejas/CNN-MNIST-MLOps-test",
            experiment_name="cnn-mnist",
            tags=["pytorch-ignite", "mnist", "MLOps", "CNN"],
            upload_source_files=["**/*.ipynb", "**/*.yaml"],
            params={'batch_size_train': args.batch_size,
                    'batch_size_test': 1000,
                    'log_interval': args.log_interval,
                    'optimizer': optimizer,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'resume_from': args.resume_from,
                    'crash_iteration': args.crash_iteration,
                    'tensorboard_logs': args.tensorboard_logs}
        )
        GradsScalarHandler = GradsScalarHandler_neptune
        WeightsScalarHandler = WeightsScalarHandler_neptune
        OptimizerParamsHandler = OptimizerParamsHandler_neptune
        global_step_from_engine = global_step_from_engine_neptune
        logger.info("Using NeptuneLogger")
    else:
        raise ValueError("Please specify either --tb_log or --neptune_log")

    # Attach the logger to the trainer to log model's weights norm after each iteration
    config_logger.attach(
        s_trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        log_handler=GradsScalarHandler(model,
                                       tag="model_trainer",
                                       reduction=torch.norm)
    )
    # Attach the logger to the trainer to log model's weights norm after each iteration
    config_logger.attach(
        s_trainer,
        event_name=Events.ITERATION_COMPLETED(every=log_interval),
        log_handler=WeightsScalarHandler(model,
                                         tag="model_trainer",
                                         reduction=torch.norm)
    )
    # Attach the logger to the trainer to log optimizer's parameters,
    # e.g. learning rate at each iteration
    config_logger.attach(
        s_trainer,
        event_name=Events.ITERATION_STARTED,
        log_handler=OptimizerParamsHandler(optimizer,
                                           tag="model_trainer")
    )
    for tag, evaluator in [("metrics/training", train_evaluator),
                           ("metrics/validation", validation_evaluator)]:
        config_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["nll", "accuracy"],
            global_step_transform=global_step_from_engine(s_trainer),
        )
    return config_logger


def trainer(model, train_loader, test_loader, args):
    """
    Trainer function.
    """
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
    tb_logs = args.tensorboard_logs
    npt_logs = args.neptune_logs

    # data:
    class_names = np.arange(10)

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
    log_dir = os.path.join(ROOT_DIR, log_dir)
    model_dir = os.path.join(ROOT_DIR, model_dir)
    write_logger = None  # init writers variables to None
    # __________________________parameters end__________________________

    # training log init:
    pbar = tqdm(initial=0,
                leave=False,
                total=len(train_loader),
                desc=f"Epoch 0 - loss: {0:.4f} - lr: {lr:.4f}")

    # Setup trainer and evaluator
    s_trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    s_trainer.logger = setup_logger("Trainer", stream=sys.stdout)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator", stream=sys.stdout)
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger("Val Evaluator", stream=sys.stdout)

    if npt_logs | tb_logs:
        # Neptune logger init:
        write_logger = logger_config(s_trainer, args, optimizer, model, log_interval,
                                     train_evaluator, validation_evaluator, log_dir)

    if early_stopping:
        # early stopping
        es_handler = EarlyStopping(
            patience=10, score_function=score_function, trainer=s_trainer
        )
        train_evaluator.add_event_handler(Events.COMPLETED, es_handler)

    if crash_iteration > 0:
        @s_trainer.on(Events.ITERATION_COMPLETED(once=crash_iteration))
        def _(engine):  # sourcery skip: raise-specific-error
            raise Exception(f"STOP at {engine.state.iteration}")

    if resume_from is not None:
        @s_trainer.on(Events.STARTED)
        def _(engine):
            pbar.n = engine.state.iteration % engine.state.epoch_length

    @s_trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        """
        Compute metrics.
        """
        pbar.refresh()
        lr_scheduler.step()  # update learning rate
        train_evaluator.run(train_loader)
        _metrics = train_evaluator.state.metrics
        avg_accuracy = _metrics["accuracy"]
        avg_nll = _metrics["nll"]
        tqdm.write(
            f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg "
            f"loss: {avg_nll:.2f} "
        )
        validation_evaluator.run(test_loader)
        _metrics = validation_evaluator.state.metrics
        avg_accuracy = _metrics["accuracy"]
        avg_nll = _metrics["nll"]
        cm = _metrics['cm']
        cm = cm.numpy()
        cm = cm.astype(int)
        cm_fig = plot_cm(class_names, cm)
        cm_fig.savefig(os.path.join(ROOT_DIR, f"reports/figures/cm_epoch_{engine.state.epoch}.png"))
        tqdm.write(
            f"Testing Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg "
            f"loss: {avg_nll:.2f} "
        )
        pbar.n = pbar.last_print_n = 0

    # Setup object to checkpoint
    objects_to_checkpoint = {
        "trainer": s_trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(model_dir, require_empty=False),
        n_saved=None,
        global_step_transform=lambda *_: s_trainer.state.epoch,
    )
    s_trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_every), training_checkpoint)

    if resume_from is not None:
        tqdm.write(f"Resume from the checkpoint: {resume_from}")
        checkpoint = torch.load(os.path.join(ROOT_DIR, resume_from))
        Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)
    try:
        # Synchronize random states
        manual_seed(SEED)
        s_trainer.run(train_loader, max_epochs=epochs)
    except Exception as _e:
        tqdm.write(f"Exception: {_e}")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        logger.info(template.format(type(_e).__name__, _e.args))
        logger.info(traceback.format_exc(), _e)

    pbar.close()
    if tb_logs | npt_logs:
        write_logger.close()
