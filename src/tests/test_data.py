import torch
from unittest import TestCase

from src.data.make_dataset import MyMNIST

dataset = MyMNIST(batch_size_train=1000)


class TestData(TestCase):
    def test_shape(self):
        """
        Test if the data has the correct shape
        """
        sample, _ = dataset.train_data[0]
        sample_test, _ = dataset.test_data[0]
        self.assertEqual(sample.shape, (1, 28, 28))
        self.assertEqual(sample_test.shape, (1, 28, 28))

    def test_scaling(self):
        """
        Test if the data has the correct scaling
        """
        for sample, _ in dataset.train_data:
            # self.assertGreaterEqual(1, sample.max())
            self.assertLessEqual(-1, sample.min())
            self.assertTrue(torch.any(sample < 0))
            self.assertTrue(torch.any(sample > 0))
        for sample, _ in dataset.test_data:
            # self.assertGreaterEqual(1, sample.max())
            self.assertLessEqual(-1, sample.min())
            self.assertTrue(torch.any(sample < 0))
            self.assertTrue(torch.any(sample > 0))

    def test_augmentation(self):
        self._check_augmentation(dataset.train_data, active=True)
        self._check_augmentation(dataset.test_data, active=False)

    def _check_augmentation(self, data, active):
        are_same = []
        for i in range(len(data)):
            sample_1, _ = data[i]
            sample_2, _ = data[i]
            are_same.append(torch.sum(sample_1 - sample_2) == 0)
        if active:
            self.assertTrue(not all(are_same))
        else:
            self.assertTrue(all(are_same))

    def test_single_process_dataloader(self):
        with self.subTest(split='train'):
            for _ in dataset.train_loader:
                pass
        with self.subTest(split='test'):
            for _ in dataset.test_loader:
                pass

    def test_multi_process_dataloader(self):
        dataset_multi_process = MyMNIST(batch_size_train=1000, num_workers=2)
        with self.subTest(split='train'):
            for _ in dataset_multi_process.train_loader:
                pass
        with self.subTest(split='test'):
            for _ in dataset_multi_process.test_loader:
                pass
