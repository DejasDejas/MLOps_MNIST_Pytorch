# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, disable=import-error, no-name-in-module
# pylint: disable=logging-fstring-interpolation
"""System module."""
import os

import torch
from torch.utils import data
from torchvision import transforms, datasets

from config.log import setup_custom_logger
from config.config import raw_path

logger = setup_custom_logger(__name__)


def load_mnist_data(batch_size_train=32, num_workers=0, split_train_data=False, split_size=0.8,
                    transform=None):
    """
    MyMNIST dataset.
    """
    logger.info("Loading dataset...")
    # create data folder:
    os.makedirs(raw_path, exist_ok=True)

    # define transforms:
    if transform is None:
        transform = transforms.ToTensor()
        test_transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.RandomRotation(5),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0.1307,), (0.3081,))])

    # load data:
    train_data = datasets.MNIST(root=raw_path,
                                train=True,
                                download=True,
                                transform=transform)
    test_data = datasets.MNIST(root=raw_path,
                               train=False,
                               download=True,
                               transform=test_transform)

    if split_train_data:
        # split data:
        train_set_size = int(len(train_data) * split_size)
        train_set, valid_set = torch.utils.data.random_split(train_data,
                                                             [train_set_size,
                                                              len(train_data) - train_set_size])
        # create data loaders:
        _train_loader = data.DataLoader(train_set,
                                        batch_size=batch_size_train,
                                        shuffle=True,
                                        num_workers=num_workers)
        _valid_loader = data.DataLoader(valid_set,
                                        batch_size=1000,
                                        shuffle=True,
                                        num_workers=num_workers)
        logger.info(f"Valid dataset size: {_valid_loader.dataset.__len__()}")
    else:
        # create data loaders:
        _train_loader = data.DataLoader(train_data,
                                        batch_size=batch_size_train,
                                        shuffle=True,
                                        num_workers=num_workers)
        _valid_loader = None

    _test_loader = data.DataLoader(test_data,
                                   batch_size=1000,
                                   shuffle=True,
                                   num_workers=num_workers)

    logger.info("Load MNIST dataset from torchvision and save to raw folder.")
    logger.info(f"Train dataset size: {_train_loader.dataset.__len__()}")
    logger.info(f"Test dataset size: {_test_loader.dataset.__len__()}")

    return _train_loader, _valid_loader, _test_loader


if __name__ == '__main__':
    # load data:
    train_loader, valid_loader, test_loader = load_mnist_data()
    print(train_loader.dataset.__len__())
    print(test_loader.dataset.__len__())
