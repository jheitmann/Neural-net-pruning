import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.layers import MaskedLinear, MaskedConv2d
from architecture.pruning_module import PruningModule


class LeNet_300_100(PruningModule):
    def __init__(self, bias=True):
        super(LeNet_300_100, self).__init__(bias)
        self.fc1 = MaskedLinear(28 * 28, 300, bias=bias)
        self.fc2 = MaskedLinear(300, 100, bias=bias)
        self.fc3 = MaskedLinear(100, 10, bias=bias)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv2(PruningModule):
    def __init__(self, bias=True):
        super(Conv2, self).__init__(bias)
        self.conv1 = MaskedConv2d(3, 64, 3, bias=bias)
        self.conv2 = MaskedConv2d(64, 64, 3, bias=bias)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = MaskedLinear(64 * 14 * 14, 256, bias=bias)
        self.fc2 = MaskedLinear(256, 256, bias=bias)
        self.fc3 = MaskedLinear(256, 10, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv4(PruningModule):
    def __init__(self, bias=True):
        super(Conv4, self).__init__(bias)
        self.conv1 = MaskedConv2d(3, 64, 3, bias=bias)
        self.conv2 = MaskedConv2d(64, 64, 3, bias=bias)
        self.conv3 = MaskedConv2d(64, 128, 3, bias=bias)
        self.conv4 = MaskedConv2d(128, 128, 3, bias=bias)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = MaskedLinear(128 * 5 * 5, 256, bias=bias)
        self.fc2 = MaskedLinear(256, 256, bias=bias)
        self.fc3 = MaskedLinear(256, 10, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv6(PruningModule):
    def __init__(self, bias=True):
        super(Conv6, self).__init__(bias)
        self.conv1 = MaskedConv2d(3, 64, 3, padding=1, bias=bias)
        self.conv2 = MaskedConv2d(64, 64, 3, padding=1, bias=bias)
        self.conv3 = MaskedConv2d(64, 128, 3, padding=1, bias=bias)
        self.conv4 = MaskedConv2d(128, 128, 3, padding=1, bias=bias)
        self.conv5 = MaskedConv2d(128, 256, 3, padding=1, bias=bias)
        self.conv6 = MaskedConv2d(256, 256, 3, padding=1, bias=bias)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = MaskedLinear(256 * 4 * 4, 256, bias=bias)
        self.fc2 = MaskedLinear(256, 256, bias=bias)
        self.fc3 = MaskedLinear(256, 10, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(PruningModule):
    def __init__(self, bias=True):
        super(AlexNet, self).__init__(bias)
        self.conv1 = MaskedConv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = MaskedConv2d(64, 192, kernel_size=3, padding=1)
        self.conv3 = MaskedConv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = MaskedConv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = MaskedConv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = MaskedLinear(256 * 2 * 2, 4096)
        self.fc2 = MaskedLinear(4096, 4096)
        self.fc3 = MaskedLinear(4096, 10)
        self.drop= nn.Dropout()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        
        x = self.drop(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.drop(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
