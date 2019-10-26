import numpy as np
import os
import torch
import torch.nn.functional as F

import common
import helpers


class Snapshots:  # add option to save all results
    def __init__(self, base_dir, model_class):
        self.base_dir = base_dir
        snapshots_path = os.path.join(base_dir, common.SNAPSHOTS_DIR)
        snapshot_fnames = [_ for _ in os.listdir(snapshots_path)]
        self.epochs = len(snapshot_fnames)
        bias = (base_dir.split('-')[1] != "unbiased")

        self.models = []
        for epoch in range(self.epochs):
            snapshot_path = os.path.join(snapshots_path, str(epoch))
            model = model_class(bias=bias)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            model_state = torch.load(snapshot_path, map_location=device)
            model.load_state_dict(model_state)
            self.models.append(model)

    def compute_fps(self, layer):  # list
        frame_potentials = []
        for model in self.models:
            layer_fp = model.compute_fp([layer])
            frame_potentials.append(layer_fp[layer])
        return frame_potentials

    def compute_ips(self, layer):  # 3-D tensor
        inner_products = []
        for model in self.models:
            layer_ips = model.compute_ips([layer])
            inner_products.append(layer_ips[layer])
        return torch.stack(inner_products)

    def compute_weight_norms(self, layer):  # matrix
        norm_series = []
        for model in self.models:
            weight_norms = model.compute_norms(layer)
            norm_series.append(weight_norms)
        return torch.stack(norm_series)

    def save_computed_metrics(self, layer):
        frame_potentials = self.compute_fps(layer)
        fp_path = helpers.train_results_path(self.base_dir, common.FP_PREFIX, layer)
        np.save(fp_path, frame_potentials)
        print("Saved frame potentials to:", fp_path)
        inner_products = self.compute_ips(layer).numpy()
        ip_path = helpers.train_results_path(self.base_dir, common.IP_PREFIX, layer)
        np.save(ip_path, inner_products)
        print("Saved inner products to:", ip_path)
        weight_norms = self.compute_weight_norms(layer).numpy()
        norms_path = helpers.train_results_path(self.base_dir, common.NORM_PREFIX, layer)
        np.save(norms_path, weight_norms)
        print("Saved weight vector norms to:", norms_path)
        return fp_path, ip_path, norms_path

    def get_weights(self, layer):
        weight_series = []
        for model in self.models:
            w = model.layer_weights(layer)
            weight_series.append(w)
        return torch.stack(weight_series)

    def get_biases(self, layer):  # matrix
        bias_series = []
        for model in self.models:
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
