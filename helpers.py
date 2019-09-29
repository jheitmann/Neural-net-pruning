import os
from datetime import datetime

import common


def model_file_path(model_id):
    date = datetime.now()
    ts = date.strftime("%d-%m-%Y_%H:%M:%S")
    model_fname = model_id + '_' + ts
    return os.path.join(common.MODEL_PATH, model_fname)


def metric_file_path(model_id, epochs, metric):
    date = datetime.now()
    ts = date.strftime("%d-%m-%Y_%H:%M:%S")
    results_fname = "{}_{}_{}_{}".format(model_id, metric, epochs, ts)
    return os.path.join(common.METRICS_PATH, results_fname)
