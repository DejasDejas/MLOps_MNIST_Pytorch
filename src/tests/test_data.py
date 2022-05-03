# pylint: disable=[invalid-name, disable=import-error, no-name-in-module, no-member]
"""System module."""
from unittest import TestCase
import torch

from src.data.make_dataset import load_mnist_data

train_loader, _, test_loader = load_mnist_data()
train_data = train_loader.dataset
test_data = test_loader.dataset


class TestData(TestCase):
    """
    Test data module.
    """
    def test_shape(self):
        """
        Test if the data has the correct shape
        """
        sample, _ = train_data[0]
        sample_test, _ = test_data[0]
        self.assertEqual(sample.shape, (1, 28, 28))
        self.assertEqual(sample_test.shape, (1, 28, 28))

    def test_scaling(self):
        """
        Test if the data has the correct scaling
        """
        for sample, _ in train_data:
            # self.assertGreaterEqual(1, sample.max())
            self.assertLessEqual(-1, sample.min())
            # self.assertTrue(torch.any(sample < 0))
            self.assertTrue(torch.any(sample > 0))
        for sample, _ in test_data:
            # self.assertGreaterEqual(1, sample.max())
            self.assertLessEqual(-1, sample.min())
            # self.assertTrue(torch.any(sample < 0))
            self.assertTrue(torch.any(sample > 0))

    def test_single_process_dataloader(self):
        """
        Test if the data has the correct augmentation
        """
        with self.subTest(split='train'):
            for _ in train_loader:
                pass
        with self.subTest(split='test'):
            for _ in test_loader:
                pass

    def test_multi_process_dataloader(self):
        """
        Test if the data has the correct augmentation
        """
        train_loader_mp, _, test_loader_mp = load_mnist_data(num_workers=2)
        with self.subTest(split='train'):
            for _ in train_loader_mp:
                pass
        with self.subTest(split='test'):
            for _ in test_loader_mp:
                pass
