import os
from datetime import datetime

import common


def log_file_path(*, log_prefix="train-logs"):
    date = datetime.now()
    log_file_name = log_prefix + '_' + date.strftime("%d-%m-%Y")
    return os.path.join(common.LOG_PATH, log_file_name)


def model_file_path(model_id):
    date = datetime.now()
    model_file_name = model_id + '_' + date.strftime("%d-%m-%Y_%H:%M:%S")
    return os.path.join(common.MODEL_PATH, model_file_name)
