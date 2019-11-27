import numpy as np
import os
import torch
import torch.nn.functional as F

import architecture.models as models
import common
import helpers
from architecture.pruning_module import inner_products


class Snapshots:  # add option to save all results
    def __init__(self, base_dir):
        self.base_dir = base_dir
        model_folder = base_dir.split('/')[-1]
        self.model_class = eval("models." + model_folder.split('-')[0])
        snapshots_path = os.path.join(base_dir, common.SNAPSHOTS_DIR)
        snapshot_fnames = [_ for _ in os.listdir(snapshots_path)]
        self.epochs = len(snapshot_fnames)
        self.bias = (base_dir.split('-')[1] != "unbiased")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.models = []
        for epoch in range(self.epochs):
            snapshot_path = os.path.join(snapshots_path, str(epoch))
            model = self.model_class(bias=self.bias)
            model_state = torch.load(snapshot_path, map_location=self.device)
            model.load_state_dict(model_state)
            self.models.append(model)

    def compute_fps(self, layer):  # list
        frame_potentials = []
        for model in self.models:
            layer_fp = model.compute_fp([layer])
            frame_potentials.append(layer_fp[layer])
        return frame_potentials

    def compute_ips(self, layer):  # 3-D tensor
        ips = []
        for model in self.models:
            layer_ips = model.compute_ips([layer])
            ips.append(layer_ips[layer])
        return torch.stack(ips)

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
        ips = self.compute_ips(layer).numpy()
        ip_path = helpers.train_results_path(self.base_dir, common.IP_PREFIX, layer)
        np.save(ip_path, ips)
        print("Saved inner products to:", ip_path)
        weight_norms = self.compute_weight_norms(layer).numpy()
        norms_path = helpers.train_results_path(self.base_dir, common.NORM_PREFIX, layer)
        np.save(norms_path, weight_norms)
        print("Saved weight vector norms to:", norms_path)
        return fp_path, ip_path, norms_path

    def create_adjacency(self, layer, merged=False):  # 3-D tensor
        result_path = helpers.prune_results_path if merged else helpers.train_results_path
        ip_path = result_path(self.base_dir, common.IP_PREFIX, layer)
        ips = np.load(ip_path)

        # Transform inner-products to distances between 0 and 1
        ips_mod = -0.5 * (ips - 1)
        kernel_width = ips_mod.mean()
        adjacency = np.exp(-ips_mod**2 / (kernel_width**2))

        # No self-loops
        for time in range(ips.shape[0]):
            for node_idx in range(ips.shape[1]):
                adjacency[time, node_idx, node_idx] = 0.

        adjacency_path = result_path(self.base_dir, common.ADJACENCY_PREFIX, layer)
        np.save(adjacency_path, adjacency)
        print("Saved adjacency matrix to:", adjacency_path)

        return adjacency, kernel_width

    def training_graph(self, layer, adjacency, merged=False):
        result_path = helpers.prune_results_path if merged else helpers.train_results_path
        norms_path = result_path(self.base_dir, common.NORM_PREFIX, layer)
        weight_norms = np.load(norms_path)
        n_epochs = adjacency.shape[0]
        n_nodes = adjacency.shape[1]

        graph = {"nodes": [], "links": []}
        for node_id in range(n_nodes):
            if weight_norms[n_epochs-1, node_id]:  # remove pruned nodes
                norms = {str(epoch): float("{:.3f}".format(weight_norms[epoch, node_id])) for epoch in range(n_epochs)}
                node_entry = {"id": str(node_id), "norm": norms}
                graph["nodes"].append(node_entry)

        for epoch in range(n_epochs):
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    edge_weight = adjacency[epoch, i, j]
                    if edge_weight:
                        edge_entry = {"source": str(i), "target": str(j), "value": float("{:.3f}".format(edge_weight)),
                                      "epoch": epoch}
                        graph["links"].append(edge_entry)

        return graph, n_epochs

    def compare_with(self, s, layer):
        w1 = s.get_weights(layer)
        w2 = self.get_weights(layer)
        merged = torch.cat((w1, w2), dim=1)
        ips = [inner_products(merged[epoch]) for epoch in range(merged.shape[0])]
        ips = torch.stack(ips).numpy()
        ip_path = helpers.prune_results_path(self.base_dir, common.IP_PREFIX, layer)
        np.save(ip_path, ips)
        print("Saved combined inner products to:", ip_path)

        weight_norms1 = s.compute_weight_norms(layer)
        weight_norms2 = self.compute_weight_norms(layer)
        weight_norms = torch.cat((weight_norms1, weight_norms2), dim=1).numpy()
        norms_path = helpers.prune_results_path(self.base_dir, common.NORM_PREFIX, layer)
        np.save(norms_path, weight_norms)
        print("Saved combined weight norms to:", norms_path)

        return ips, weight_norms

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
        ips = w_mod.matmul(w.t())
        return ips

    def sub_network(self, layer, pruned_indices):  # set initial mask only -> take into account when plotting
        snapshot_path = os.path.join(self.base_dir, common.SNAPSHOTS_DIR, '0')
        model = self.model_class(bias=self.bias)
        model_state = torch.load(snapshot_path, map_location=self.device)
        model.load_state_dict(model_state)
        for idx in pruned_indices:
            model.prune_element(layer, idx)
        return model
