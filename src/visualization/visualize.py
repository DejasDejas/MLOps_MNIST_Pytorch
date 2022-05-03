# pylint: disable=[invalid-name, disable=import-error, no-name-in-module]
"""System module."""
import matplotlib.pyplot as plt
import torch
from src.data.make_dataset import MyMNIST


def sample_data_visualization(dataloader, n_samples=10, n_cols=5):
    """
    Visualization of sample data from the data set.

    :param dataloader: dataloader object
    :param n_samples: the number of samples
    :param n_cols: the number of columns
    :return: the sampled data
    """
    figure = plt.figure(figsize=(10, 8))
    n_rows = n_samples // n_cols
    examples = enumerate(dataloader)
    _, (example_data, example_targets) = next(examples)
    for i in range(1, n_cols * n_rows + 1):
        sample_idx = torch.randint(len(example_data), size=(1,)).item()
        img = example_data[sample_idx]
        label = example_targets[sample_idx].item()
        figure.add_subplot(n_rows, n_cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


if __name__ == '__main__':
    dataset = MyMNIST()
    sample_data_visualization(dataset.train_data, n_samples=25, n_cols=5)
