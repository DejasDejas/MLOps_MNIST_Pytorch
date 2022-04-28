from pathlib import Path
from argparse import ArgumentParser
import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from ignite.engine import (Events,
                           create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.metrics import Accuracy, Loss
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

from src.data.data_loader import mnist_dataloader
from src.models.model import CNN
from src.models.utils import (
    log_model_weights,
    log_data_stats,
    log_model_grads,
    plot_classes_preds,
    gpu_config
)
import config.log as log
from config.config import ROOT_DIR

logger = log.setup_custom_logger(__name__)

SEED = 1


def score_function(engine):
    val_loss = engine.state.metrics["nll"]
    return -val_loss


def main(args):
    # arguments parameters:
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    epochs = args.epochs
    lr = args.lr
    log_interval = args.log_interval
    log_dir = args.log_dir + datetime.datetime.now().strftime("%Y_%m_%d--%Hh%Mmn%S")
    model_dir = args.model_dir
    checkpoint_every = args.checkpoint_every
    resume_from = args.resume_from
    crash_iteration = args.crash_iteration

    # data:
    class_names = np.arange(10)
    train_loader, valid_loader, test_loader = mnist_dataloader(batch_size,
                                                               val_batch_size)

    # training log configuration:
    log_dir = os.path.join(ROOT_DIR, log_dir)
    model_dir = os.path.join(ROOT_DIR, model_dir)
    writer = SummaryWriter(log_dir=log_dir)
    pbar = tqdm(initial=0,
                leave=False,
                total=len(train_loader),
                desc=f"Epoch 0 - loss: {0:.4f} - lr: {lr:.4f}",
                )

    # model and training configuration:
    model = CNN()
    model, device = gpu_config(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }

    # inspect the model using TensorBoard
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    writer.add_graph(model, images)

    # Projector to TensorBoard
    features = images.view(-1, 1 * 28 * 28)
    writer.add_embedding(features, metadata=labels, label_img=images)

    # Setup trainer and evaluator
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    # early stopping
    es_handler = EarlyStopping(
        patience=10, score_function=score_function, trainer=trainer
    )

    # Apply learning rate scheduling
    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_step():
        lr_scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        _lr = optimizer.param_groups[0]["lr"]
        pbar.desc = f"Epoch {engine.state.epoch} - loss: {engine.state.output:.4f} - lr: {_lr:.4f}"
        pbar.update(log_interval)
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
        writer.add_scalar("lr", _lr, engine.state.iteration)

    if crash_iteration > 0:
        @trainer.on(Events.ITERATION_COMPLETED(once=crash_iteration))
        def _(engine):
            raise Exception(f"STOP at {engine.state.iteration}")

    if resume_from is not None:
        @trainer.on(Events.STARTED)
        def _(engine):
            pbar.n = engine.state.iteration % engine.state.epoch_length

    @trainer.on(Events.EPOCH_COMPLETED)
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

    # Compute and tensorboard_logs validation metrics
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_loader)
        _metrics = evaluator.state.metrics
        avg_accuracy = _metrics["accuracy"]
        avg_nll = _metrics["nll"]
        evaluator.add_event_handler(Events.COMPLETED, es_handler)
        tqdm.write(
            f"Test Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.2f} Avg loss: {avg_nll:.2f}"
        )
        pbar.n = pbar.last_print_n = 0
        writer.add_scalar("validation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)

    # Setup object to checkpoint
    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(model_dir, require_empty=False),
        n_saved=None,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(
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
        trainer.add_event_handler(
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
        manual_seed(15)
        trainer.run(train_loader, max_epochs=epochs)
    except Exception as _e:
        import traceback

        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        logger.info(template.format(type(_e).__name__, _e.args))
        logger.info(traceback.format_exc(), _e)

    # _____________________________________ Prediction on test dataloader __________________________________________
    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the predictions in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_label = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            output = model(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(labels)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)

    # helper function
    def add_pr_curve_tensorboard(class_index, _test_probs, _test_label, global_step=0):
        """
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        """
        tensorboard_truth = _test_label == class_index
        tensorboard_probs = _test_probs[:, class_index]

        writer.add_pr_curve(
            str(class_names[class_index]),
            tensorboard_truth,
            tensorboard_probs,
            global_step=global_step,
        )

    # plot all the pr curves
    for i in range(len(class_names)):
        add_pr_curve_tensorboard(i, test_probs, test_label)

    # _____________________________________ Fin Prediction on test dataloader ________________________________________

    @trainer.on(Events.EPOCH_COMPLETED | Events.COMPLETED)
    def log_time():
        tqdm.write(
            f"{trainer.last_event_name.name} took {trainer.state.times[trainer.last_event_name.name]} seconds"
        )

    pbar.close()
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training configuration for MNIST classification")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=1000,
        help="input batch size for validation (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=2,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="reports/tensorboard/logs/",
        help="tensorboard_logs directory for " "Tensorboard tensorboard_logs output ",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/",
        help="model directory for save models trained",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5,
        help="Checkpoint training every X epochs",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to the checkpoint .pt file to resume training from ",
    )
    parser.add_argument(
        "--crash_iteration",
        type=int,
        default=-1,
        help="Iteration at which to raise an exception",
    )

    arguments = parser.parse_args()
    main(arguments)
