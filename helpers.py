import os
from datetime import datetime

import common


def model_file_path(model_id):
    date = datetime.now()
    ts = date.strftime("%d-%m-%Y_%H:%M:%S")
    model_fname = model_id + '_' + ts
    return os.path.join(common.MODEL_PATH, model_fname)


def train_results_path(model_id, epochs, metric, *, layer=""):
    date = datetime.now()
    ts = date.strftime("%d-%m-%Y_%H:%M:%S")
    if layer:
        results_fname = "{}_{}e_{}_{}_{}".format(model_id, epochs, metric, layer, ts)
    else:
        results_fname = "{}_{}e_{}_{}".format(model_id, epochs, metric, ts)
    return os.path.join(common.TRAIN_METRICS_PATH, results_fname)


def prune_results_path(model_id, layer, pruning_method, pruning_ratio, metric):
    date = datetime.now()
    ts = date.strftime("%d-%m-%Y_%H:%M:%S")
    results_fname = "{}_{}_{}_{}_{}_{}".format(model_id, layer, pruning_method, pruning_ratio, metric, ts)
    return os.path.join(common.PRUNE_METRICS_PATH, results_fname)
