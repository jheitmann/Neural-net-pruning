import torch
import torch.nn as nn
import torch.nn.functional as F

import common
import helpers
from layers import MaskedLinear, MaskedConv2d


def inner_products(w, normalize=True):
    if normalize:
        w = F.normalize(w, p=2, dim=1)  # normalized weight rows
    ips = w.matmul(w.t())
    return ips


def frame_potential(ips, N):
    inner_product_sum = ips.matmul(ips).trace().item()
    fp = inner_product_sum / (N ** 2)
    return fp


def mean_inner_product(ips, N):  # not really mean inner product (zero weights)
    abs_ip_sum = torch.abs(ips).sum().item()
    mean_abs_ip = abs_ip_sum / (N ** 2)
    return mean_abs_ip


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class PruningModule(nn.Module):
    def __init__(self):
        super(PruningModule, self).__init__()

    def model_id(self):
        return self.__class__.__name__

    def compute_fp(self, monitored):
        layer_ips = {}
        layer_fp = {}
        with torch.no_grad():
            for layer in monitored:
                param = getattr(self, layer)
                w = param.get_weights()
                ips = inner_products(w)  # add normalize=True?
                fp = frame_potential(ips, w.shape[0])
                layer_ips[layer] = ips
                layer_fp[layer] = fp
        return layer_ips, layer_fp

    def compute_mean_ip(self, monitored):  # changeme
        layer_mean_ip = {}
        with torch.no_grad():
            for layer in monitored:
                param = getattr(self, layer)
                w = param.get_weights()
                mean_abs_ip = mean_inner_product(w, w.shape[0])
                layer_mean_ip[layer] = mean_abs_ip
        return layer_mean_ip

    def selective_correlation(self, layer, l2_norm=True, normalize=True):
        partial_corrs = []
        corr_metric = frame_potential if l2_norm else mean_inner_product
        with torch.no_grad():
            param = getattr(self, layer)
            w = param.get_weights()
            unpruned_indices = param.unpruned_parameters()
            for i in unpruned_indices:
                indices = [j for j in unpruned_indices if j != i]
                all_but_one = torch.index_select(w, 0, w.new_tensor(indices, dtype=torch.long))
                ips = inner_products(all_but_one, normalize)
                corr = corr_metric(ips, w.shape[0])
                partial_corrs.append((corr, i))
        return partial_corrs

    def compute_squared_norms(self, layer):
        with torch.no_grad():
            param = getattr(self, layer)
            w = param.get_weights()
            unpruned_indices = param.unpruned_parameters()
            squared_norms = [(w[i].matmul(w[i].t()), i) for i in unpruned_indices]
        return squared_norms

    def prune_element(self, layer, pruning_idx):
        with torch.no_grad():
            param = getattr(self, layer)
            w = param.get_weights()
            rows, cols = w.shape
            mask = param.get_mask()
            mask[pruning_idx] = torch.zeros(cols)
            param.set_mask(mask)


class LeNet_300_100(PruningModule):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        self.fc1 = MaskedLinear(28 * 28, 300)
        self.fc2 = MaskedLinear(300, 100)
        self.fc3 = MaskedLinear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv2(PruningModule):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = MaskedConv2d(3, 64, 3)
        self.conv2 = MaskedConv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = MaskedLinear(64 * 14 * 14, 256)
        self.fc2 = MaskedLinear(256, 256)
        self.fc3 = MaskedLinear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvTest(PruningModule):
    def __init__(self):
        super(ConvTest, self).__init__()
        self.conv1 = MaskedConv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = MaskedConv2d(6, 16, 5)
        self.fc1 = MaskedLinear(16 * 5 * 5, 120)
        self.fc2 = MaskedLinear(120, 84)
        self.fc3 = MaskedLinear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
