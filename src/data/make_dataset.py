# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.utils.data as data
import os

import config.log as log
from config.config import raw_path

logger = log.setup_custom_logger(__name__)


class MyMNIST(data.Dataset):
    def __init__(self, batch_size_train=32, batch_size_test=1000, num_workers=0, split_train_data=False, split_size=0.8,
                 transform=None):

        # parameters:
        self.train_set = None
        self.split_size = split_size
        self.transform = transform
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.split_train_data = split_train_data

        # create data folder:
        os.makedirs(raw_path, exist_ok=True)

        # define transforms:
        if self.transform is None:
            self.transform = torchvision.transforms.ToTensor()
            self.test_transform = torchvision.transforms.ToTensor()
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomRotation(5),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))])
            self.test_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])

        # load data:
        self.train_data = torchvision.datasets.MNIST(root=raw_path,
                                                     train=True,
                                                     download=True,
                                                     transform=self.transform)
        self.test_data = torchvision.datasets.MNIST(root=raw_path,
                                                    train=False,
                                                    download=True,
                                                    transform=self.test_transform)

        if self.split_train_data:
            # split data:
            self.train_set_size = int(len(self.train_set) * split_size)
            self.valid_set_size = len(self.train_set) - self.train_set_size
            self.train_set, self.valid_set = torch.utils.data.random_split(self.train_set,
                                                                           [self.train_set_size,
                                                                            self.valid_set_size])

            # create data loaders:
            self.train_loader = data.DataLoader(self.train_set,
                                                batch_size=self.batch_size_train,
                                                shuffle=True,
                                                num_workers=self.num_workers)
            self.valid_loader = data.DataLoader(self.valid_set,
                                                batch_size=self.batch_size_test,
                                                shuffle=True,
                                                num_workers=self.num_workers)
            logger.info(f'Valid dataset size: {len(self.valid_loader.dataset)}')
        else:
            # create data loaders:
            self.train_loader = data.DataLoader(self.train_data,
                                                batch_size=self.batch_size_train,
                                                shuffle=True,
                                                num_workers=self.num_workers)

        self.test_loader = data.DataLoader(self.test_data,
                                           batch_size=self.batch_size_test,
                                           shuffle=True,
                                           num_workers=self.num_workers)

        logger.info('Load MNIST dataset from torchvision and save to raw folder.')
        logger.info(f'Train dataset size: {len(self.train_loader.dataset)}')
        logger.info(f'Test dataset size: {len(self.test_loader.dataset)}')


if __name__ == '__main__':
    data = MyMNIST(batch_size_train=1000)
    print(len(data.train_loader.dataset))
    print(len(data.test_loader.dataset))
