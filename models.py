import torch
import torch.nn as nn
import torch.nn.functional as F

import common
from layers import MaskedLinear, MaskedConv2d


def squared_norm(w):
    squared_norms = w.matmul(w.t())
    return squared_norms


def frame_potential(w, N):
    w_normalized = F.normalize(w, p=2, dim=1)  # normalized weight rows
    
    T_mod = w_normalized.matmul(w_normalized.t())
    inner_product_sum = T_mod.matmul(T_mod).trace().item()  

    fp = inner_product_sum / (N**2)
    return fp


class PruningModule(nn.Module):
    def __init__(self):
        super(PruningModule, self).__init__()


    def model_ID(self):
        return self.__class__.__name__
    

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
    def modified_dp(self, monitored):
        pass  # sum(abs(w @ w.T))


    def compute_fp(self, monitored):
        layer_fp = {}
        with torch.no_grad():
            for layer in monitored:
                param = getattr(self, layer)
                w = param.get_weights()
                fp = frame_potential(w, w.shape[0])
                layer_fp[layer] = fp
        return layer_fp


    def selective_fps(self, layer):
        partial_fps = []
        with torch.no_grad():
            param = getattr(self, layer)
            w = param.get_weights()
            unpruned_indices = param.unpruned_parameters()
            for i in unpruned_indices:
                indices = [j for j in unpruned_indices if j != i]
                all_but_one = torch.index_select(w, 0, w.new_tensor(indices, dtype=torch.long))
                fp = frame_potential(all_but_one, w.shape[0])
                partial_fps.append((fp, i))
        return partial_fps


    def compute_squared_norms(self, layer):
        squared_norms = []
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

    
    def prune_layer(self, layer, pruning, pruning_iters):
        for pruning_iter, pruning_idx in enumerate(pruning(self, layer, pruning_iters)):
            self.prune_element(layer, pruning_idx)


class LeNet_300_100(PruningModule):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        self.fc1 = MaskedLinear(28*28, 300)
        self.fc2 = MaskedLinear(300, 100)
        self.fc3 = MaskedLinear(100, 10)


    def forward(self, x):
        x = x.view(-1, 28*28)
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
        