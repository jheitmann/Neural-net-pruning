import os
import torch


class Snapshots:
    def __init__(self, dir_name, epochs, Model):  # epochs is a list
        self.models = {}
        for epoch in epochs:
            snapshot_fname = os.path.join(dir_name, str(epoch))
            model = Model()
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
            inner_products.append(layer_ips)
        return torch.stack(inner_products)

    def compute_weight_norms(self, layer):  # matrix
        weight_norms = []
        for epoch in self.epochs:
            model = self.models[epoch]
            norms = model.compute_squared_norms(layer)
            weight_norms.append(norms)
        return torch.stack(weight_norms, dim=1)

    def get_biases(self, layer):  # matrix
        pass
