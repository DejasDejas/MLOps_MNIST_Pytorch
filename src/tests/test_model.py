# pylint: disable=[invalid-name, disable=import-error, no-name-in-module, no-member, not-callable]
"""System module."""
import unittest
import torch

from src.models.model import CNN


class TestCNN(unittest.TestCase):
    """
    Test CNN class.
    """
    def setUp(self) -> None:
        """
        Set up the test.
        """
        self.net = CNN()
        self.inputs = torch.rand(4, 1, 28, 28)

    def test_shape(self):
        """
        Test shape of the output.
        """
        net = self.net
        x = self.inputs
        y = net(x)
        self.assertEqual(torch.Size((4, 10,)), y.shape)

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_device_moving(self):
        """
        Test if the model is moved to the GPU.
        """
        net = self.net
        net_on_gpu = net.to('cuda:0')
        net_back_on_cpu = net_on_gpu.cpu()
        inputs = self.inputs
        torch.manual_seed(42)
        outputs_cpu = net(inputs)
        torch.manual_seed(42)
        outputs_gpu = net_on_gpu(inputs.to('cuda:0'))
        torch.manual_seed(42)
        outputs_back_on_cpu = net_back_on_cpu(inputs)
        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_gpu.cpu()))
        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_back_on_cpu))

    def test_all_parameters_updated(self):
        """
        Test if all parameters are updated.
        """
        net = self.net
        optim = torch.optim.SGD(net.parameters(), lr=0.1)

        outputs = net(self.inputs)
        loss = outputs.mean()
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    self.assertNotEqual(0., torch.sum(param.grad ** 2))
