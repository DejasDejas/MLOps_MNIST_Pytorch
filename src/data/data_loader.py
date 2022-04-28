# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.utils.data as data

import config.log as log
from config.config import raw_path

logger = log.setup_custom_logger(__name__)


def mnist_dataloader(batch_size_train=64, batch_size_test=1000, split_size=0.8):
    """
    Load Mnist Dataset from Torchvision datasets.
    """
    train_set = torchvision.datasets.MNIST(raw_path,
                                           train=True,
                                           download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))
    # Random split train and valid set
    train_set_size = int(len(train_set) * split_size)
    valid_set_size = len(train_set) - train_set_size
    train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

    train_loader = data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size_test, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(raw_path,
                                   train=False,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    logger.info('Load MNIST dataset from torchvision and save to raw folder.')
    logger.info(f'Train dataset size: {len(train_loader.dataset)}')
    logger.info(f'Valid dataset size: {len(valid_loader.dataset)}')
    logger.info(f'Test dataset size: {len(test_loader.dataset)}')
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    _, _, _ = mnist_dataloader()
