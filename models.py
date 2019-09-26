import torch
import torch.nn as nn
import torch.nn.functional as F

import common
from layers import MaskedLinear


# Seeding for reproducibility
torch.manual_seed(common.SEED)

"""
General remark: add detach()
"""


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


    def compute_fp(self, monitored):  # use unpruned indices?
        layer_fp = {}
        with torch.no_grad():
            for layer in monitored:
                param = getattr(self, layer)
                w = param.get_weights()
                N = len(param.unpruned_parameters())
                fp = frame_potential(w, N)
                layer_fp[layer] = fp
        return layer_fp


    def selective_fps(self, layer):  # add if mask
        partial_fps = []
        with torch.no_grad():  # not necessary?
            param = getattr(self, layer)
            w = param.get_weights()
            unpruned_indices = param.unpruned_parameters()
            N = len(unpruned_indices)
            for i in unpruned_indices:
                indices = [j for j in unpruned_indices if j != i]
                all_but_one = torch.index_select(w, 0, w.new_tensor(indices, dtype=torch.long))
                fp = frame_potential(all_but_one, N-1)
                partial_fps.append((fp, i))
        return partial_fps

    
    def prune_element(self, layer, pruning_idx):  # test function
        with torch.no_grad():  # not necessary?
            param = getattr(self, layer)
            w = param.get_weights()
            rows, cols = w.shape
            mask = param.mask if param.mask_flag else w.new_full(w.shape, 1.)  # should have same device
            mask[pruning_idx] = torch.zeros(cols)
            param.set_mask(mask)


class LeNet_300_100(PruningModule):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)


    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet_300_100_Pruned(PruningModule):
    def __init__(self):
        super(LeNet_300_100_Pruned, self).__init__()
        self.fc1 = MaskedLinear(28*28, 300)
        self.fc2 = MaskedLinear(300, 100)
        self.fc3 = MaskedLinear(100, 10)


    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet_300_100_SELU(nn.Module):
    def __init__(self):
        super(LeNet_300_100_SELU, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)


    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x


class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
