import json
import os
import torch

import common
import helpers
from architecture.models import VGG19_BN
from processing.snapshots import Snapshots
from vgg_guille import VGG


snapshot_path = "out/VGG_Guille/snapshots/0"
model_state = torch.load(snapshot_path, map_location=torch.device("cpu"))["model"]
model = VGG19_BN()

features = list(model_state.keys())[:-6]
classifier = list(model_state.keys())[-6:]
conv_chuncks = [features[i:i+7] for i in range(0, len(features), 7)]
fc_chuncks = [classifier[i:i+2] for i in range(0, len(classifier), 2)]

# [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# ['features.0.weight', 'features.0.bias']
# ['features.1.weight', 'features.1.bias', 'features.1.running_mean', 'features.1.running_var', 'features.1.num_batches_tracked']

with torch.no_grad():
    for i, chunk in enumerate(conv_chuncks):
        conv_layer = f"conv{i+1}"
        bn_layer = f"bn{i+1}"
        conv_param = getattr(model, conv_layer)
        bn_param = getattr(model, bn_layer)
        conv_param.weight.data = model_state[chunk[0]]
        conv_param.bias.data = model_state[chunk[1]]
        bn_param.weight.data = model_state[chunk[2]]
        bn_param.bias.data = model_state[chunk[3]]
        bn_param.running_mean.data = model_state[chunk[4]]
        bn_param.running_var.data = model_state[chunk[5]]
        bn_param.num_batches_tracked.data = model_state[chunk[6]]
    for i, chunk in enumerate(fc_chuncks):
        fc_layer = f"fc{i+1}"
        fc_param = getattr(model, fc_layer)
        fc_param.weight.data = model_state[chunk[0]]
        fc_param.bias.data = model_state[chunk[1]]

base_dir, dir_paths = helpers.model_results_path(model.model_id())
snapshot_dir = dir_paths[common.SNAPSHOTS_DIR]
snapshot_fname = os.path.join(snapshot_dir, '0')
torch.save(model.state_dict(), snapshot_fname)

s = Snapshots(base_dir)
fp_paths = {}
ip_paths = {}
norms_paths = {}
layers = [f"conv{i}" for i in range(1, 17)]
layers += ["fc1", "fc2", "fc3"]
for layer in layers:
    fp_path, ip_path, norms_path = s.save_computed_metrics(layer)
    fp_paths[layer] = fp_path
    ip_paths[layer] = ip_path
    norms_paths[layer] = norms_path

with open(common.MODEL_SPECS_PATH, 'r') as rfp:
    models = json.load(rfp)

models[base_dir] = layers
with open(common.MODEL_SPECS_PATH, 'w') as wfp:
    json.dump(models, wfp, sort_keys=True, indent=4)
