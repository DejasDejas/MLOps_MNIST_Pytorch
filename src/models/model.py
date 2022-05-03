# pylint: disable=[invalid-name, disable=import-error, no-name-in-module]
"""System module."""
import torch.nn.functional as F
import torch.nn as nn
from src.models.custom_layer import kaiming_init


class CNN(nn.Module):
    """
    CNN model.
    """
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Sequential(nn.Linear(32 * 7 * 7, 10))

        self.weight_init()

    def forward(self, x):
        """
        Forward pass.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return F.log_softmax(output, dim=1)

    def weight_init(self):
        """
        Weight initialization.
        """
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


if __name__ == "__main__":
    model = CNN()
    print(model)
