import math

import needle as ndl
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from cifar import train_cifar10
from models.resnet9 import ResNet9

from tests.utils import set_random_seeds

_DEVICES = [
    ndl.cpu(),
    pytest.param(
        ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")
    ),
]

rng = np.random.default_rng(0)


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PytorchResNet9(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 10,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Define all layers explicitly instead of using _make_conv_layer helper
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # First residual block
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Second residual block
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, out_features)

        # Move to specified device if provided
        if device is not None:
            self.to(device)

        # Set dtype if specified
        if dtype is not None:
            if dtype == "float32":
                self.to(torch.float32)
            elif dtype == "float16":
                self.to(torch.float16)

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))

        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))

        # First residual block
        residual = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x + residual  # Residual connection

        # Third conv block
        x = F.relu(self.bn5(self.conv5(x)))

        # Fourth conv block
        x = F.relu(self.bn6(self.conv6(x)))

        # Second residual block
        residual = x
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = x + residual  # Residual connection

        # Flatten and FC layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_resnet9_number_of_parameters(device):
    def num_params(model):
        return sum([math.prod(x.shape) for x in model.parameters()])

    model = ResNet9(device=device)

    assert num_params(model) == 431946


@pytest.mark.skip(reason="This test does not work yet")
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_resnet9_first_epoch(device):
    set_random_seeds(0)

    model = ResNet9(device=device)
    torch_model = PytorchResNet9()

    a = np.random.randn(2, 3, 32, 32).astype(np.float32)
    A = ndl.Tensor(a, device=device)

    y = model(A)

    for m in torch_model.modules():
        if isinstance(m, (torch.nn.Conv2d | torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_in")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    torch_y = torch_model(torch.tensor(a)).detach().numpy()

    np.testing.assert_allclose(y.numpy(), torch_y, atol=1e-6)


@pytest.mark.skip(reason="This test is extremely slow so far")
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.slow
def test_train_cifar10(device, atol=1e-2):
    set_random_seeds(0)

    dataset = ndl.data.CIFAR10Dataset(train=True)
    dataloader = ndl.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
    )

    model = ResNet9(device=device, dtype="float32")
    acc, loss = train_cifar10(
        dataloader=dataloader,
        model=model,
        n_epochs=1,
        loss_fn=ndl.nn.SoftmaxLoss,
        optimizer=ndl.optim.Adam,
    )
    print(f"Train accuracy: {acc}, loss: {loss}")
    np.testing.assert_allclose(
        acc,
        0.09375,
        atol=atol,
        err_msg="Accuracy is not within the expected range.",
    )
    np.testing.assert_allclose(
        loss,
        3.5892258,
        atol=atol,
        err_msg="Loss is not within the expected range.",
    )
