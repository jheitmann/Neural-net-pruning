import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.layers import MaskedLinear, MaskedConv2d


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
    def __init__(self, bias):
        super(PruningModule, self).__init__()
        self.bias = bias

    def model_id(self):
        name = self.__class__.__name__
        if not self.bias:
            name = '-'.join((name, "unbiased"))
        return name

    def compute_ips(self, monitored):
        layer_ips = {}
        with torch.no_grad():
            for layer in monitored:
                param = getattr(self, layer)
                w = param.get_weights()
                ips = inner_products(w)  # add normalize=True?
                layer_ips[layer] = ips
        return layer_ips

    def compute_fp(self, monitored):
        layer_ips = self.compute_ips(monitored)
        layer_fp = {layer: frame_potential(ips, ips.shape[0]) for layer, ips in layer_ips.items()}
        return layer_fp

    def compute_mean_ip(self, monitored):
        layer_ips = self.compute_ips(monitored)
        layer_mean_ip = {layer: mean_inner_product(ips, ips.shape[0]) for layer, ips in layer_ips.items()}
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

    def compute_norms(self, layer):
        with torch.no_grad():
            param = getattr(self, layer)
            w = param.get_weights()
            weight_norms = w.norm(dim=1)
        return weight_norms

    def layer_weights(self, layer):
        param = getattr(self, layer)
        w = param.get_weights()
        return w

    def layer_biases(self, layer):
        param = getattr(self, layer)
        b = param.get_biases()
        return b

    def prune_element(self, layer, pruning_idx):
        with torch.no_grad():
            param = getattr(self, layer)
            w = param.get_weights()
            rows, cols = w.shape
            mask = param.get_mask()
            mask[pruning_idx] = torch.zeros(cols)
            param.set_mask(mask)


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


class LeNet_20_100(PruningModule):
    def __init__(self, bias=True):
        super(LeNet_20_100, self).__init__(bias)
        self.fc1 = MaskedLinear(28 * 28, 20, bias=bias)
        self.fc2 = MaskedLinear(20, 100, bias=bias)
        self.fc3 = MaskedLinear(100, 10, bias=bias)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet_300_10(PruningModule):
    def __init__(self, bias=True):
        super(LeNet_300_10, self).__init__(bias)
        self.fc1 = MaskedLinear(28 * 28, 300, bias=bias)
        self.fc2 = MaskedLinear(300, 10, bias=bias)
        self.fc3 = MaskedLinear(10, 10, bias=bias)

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


class Conv2_30(PruningModule):
    def __init__(self, bias=True):
        super(Conv2_30, self).__init__(bias)
        self.conv1 = MaskedConv2d(3, 64, 3, bias=bias)
        self.conv2 = MaskedConv2d(64, 64, 3, bias=bias)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = MaskedLinear(64 * 14 * 14, 30, bias=bias)
        self.fc2 = MaskedLinear(30, 256, bias=bias)
        self.fc3 = MaskedLinear(256, 10, bias=bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
