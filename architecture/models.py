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
        self.conv1 = MaskedConv2d(3, 64, 3, padding=1, bias=bias)  # modified
        self.conv2 = MaskedConv2d(64, 64, 3, padding=1, bias=bias)  # modified
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = MaskedLinear(64 * 16 * 16, 256, in_channels=64, bias=bias)  # modified
        self.fc2 = MaskedLinear(256, 256, bias=bias)
        self.fc3 = MaskedLinear(256, 10, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # modified
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv4(PruningModule):
    def __init__(self, bias=True):
        super(Conv4, self).__init__(bias)
        self.conv1 = MaskedConv2d(3, 64, 3, padding=1, bias=bias)  # modified
        self.conv2 = MaskedConv2d(64, 64, 3, padding=1, bias=bias)  # modified
        self.conv3 = MaskedConv2d(64, 128, 3, padding=1, bias=bias)  # modified
        self.conv4 = MaskedConv2d(128, 128, 3, padding=1, bias=bias)  # modified
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = MaskedLinear(128 * 8 * 8, 256, in_channels=128, bias=bias)
        self.fc2 = MaskedLinear(256, 256, bias=bias)
        self.fc3 = MaskedLinear(256, 10, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 128 * 8 * 8)
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

        self.fc1 = MaskedLinear(256 * 4 * 4, 256, in_channels=256, bias=bias)
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


class VGG11(PruningModule):

    def __init__(self, bias=True, init_weights=False):
        super(VGG11, self).__init__(bias)

        self.conv1 = MaskedConv2d(3, 64, kernel_size=3, padding=1, bias=bias)
        self.conv2 = MaskedConv2d(64, 128, kernel_size=3, padding=1, bias=bias)
        self.conv3 = MaskedConv2d(128, 256, kernel_size=3, padding=1, bias=bias)
        self.conv4 = MaskedConv2d(256, 256, kernel_size=3, padding=1, bias=bias)
        self.conv5 = MaskedConv2d(256, 512, kernel_size=3, padding=1, bias=bias)
        self.conv6 = MaskedConv2d(512, 512, kernel_size=3, padding=1, bias=bias)
        self.conv7 = MaskedConv2d(512, 512, kernel_size=3, padding=1, bias=bias)
        self.conv8 = MaskedConv2d(512, 512, kernel_size=3, padding=1, bias=bias)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = MaskedLinear(512 * 7 * 7, 4096, bias=bias)
        self.fc2 = MaskedLinear(4096, 4096, bias=bias)
        self.fc3 = MaskedLinear(4096, 10, bias=bias)
        self.drop = nn.Dropout()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = self.pool(F.relu(self.conv8(x)))

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MaskedLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG19_BN(PruningModule):

    def __init__(self, bias=True):
        super(VGG19_BN, self).__init__(bias)

        self.conv1 = MaskedConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = MaskedConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = MaskedConv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = MaskedConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = MaskedConv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = MaskedConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = MaskedConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = MaskedConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = MaskedConv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = MaskedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = MaskedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = MaskedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = MaskedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = MaskedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = MaskedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = MaskedConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn16 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = MaskedLinear(512, 512, in_channels=512, bias=bias)
        self.fc2 = MaskedLinear(512, 512, bias=bias)
        self.fc3 = MaskedLinear(512, 10, bias=bias)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = self.pool(x)

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))

        x = self.pool(x)

        x = self.conv9(x)
        x = F.relu(self.bn9(x))
        x = self.conv10(x)
        x = F.relu(self.bn10(x))
        x = self.conv11(x)
        x = F.relu(self.bn11(x))
        x = self.conv12(x)
        x = F.relu(self.bn12(x))

        x = self.pool(x)

        x = self.conv13(x)
        x = F.relu(self.bn13(x))
        x = self.conv14(x)
        x = F.relu(self.bn14(x))
        x = self.conv15(x)
        x = F.relu(self.bn15(x))
        x = self.conv16(x)
        x = F.relu(self.bn16(x))

        x = self.pool(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
