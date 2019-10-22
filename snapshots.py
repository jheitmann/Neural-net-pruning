import os
import torch
import torch.nn.functional as F


class Snapshots:
    def __init__(self, dir_name, epochs, model_class, bias=True):  # epochs is a list
        self.models = {}
        self.epochs = epochs
        for epoch in epochs:
            snapshot_fname = os.path.join(dir_name, str(epoch))
            model = model_class(bias=bias)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model_state = torch.load(snapshot_fname, map_location=device)
            model.load_state_dict(model_state)
            self.models[epoch] = model

    def compute_fps(self, layer):  # list
        frame_potentials = []
        for epoch in self.epochs:
            model = self.models[epoch]
            _, layer_fp = model.compute_fp([layer])
            frame_potentials.append(layer_fp[layer])
        return frame_potentials

    def compute_ips(self, layer):  # 3-D tensor
        inner_products = []
        for epoch in self.epochs:
            model = self.models[epoch]
            layer_ips, _ = model.compute_fp([layer])
            inner_products.append(layer_ips[layer])
        return torch.stack(inner_products)

    def compute_weight_norms(self, layer):  # matrix
        norm_series = []
        for epoch in self.epochs:
            model = self.models[epoch]
            weight_norms = model.compute_norms(layer)
            norm_series.append(weight_norms)
        return torch.stack(norm_series)

    def get_weights(self, layer):
        weight_series = []
        for epoch in self.epochs:
            model = self.models[epoch]
            w = model.layer_weights(layer)
            weight_series.append(w)
        return torch.stack(weight_series)

    def get_biases(self, layer):  # matrix
        bias_series = []
        for epoch in self.epochs:
            model = self.models[epoch]
            biases = model.layer_biases(layer)
            bias_series.append(biases)
        return torch.stack(bias_series)

    def compute_ips_with(self, layer, w_mod, epoch):
        model = self.models[epoch]
        w = model.layer_weights(layer)
        w = F.normalize(w, p=2, dim=1)
        w_mod = F.normalize(w_mod, p=2, dim=1)
        inner_products = w_mod.matmul(w.t())
        return inner_products
