from argparse import ArgumentParser
from src.trainer.utils import str2bool
from src.data.make_dataset import MyMNIST
from src.models.model import CNN
from src.trainer.trainer import trainer


def run(args):
    # data:
    dataset = MyMNIST(batch_size_train=args.batch_size)
    # model:
    model = CNN()
    # train:
    trainer(model, dataset, args)

    return


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
        "--epochs", type=int, default=1, help="number of epochs to train (default: 10)"
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
    parser.add_argument(
        "--tb_graph_model",
        default=True,
        type=str2bool,
        help="True if you want to save the graph model on TensorBoard.",
    )
    parser.add_argument(
        "--early_stopping",
        default=False,
        type=str2bool,
        help="True if you want to use early stopping.",
    )

    arguments = parser.parse_args()
    run(arguments)
