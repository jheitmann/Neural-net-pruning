import os
from datetime import datetime

import common


def log_file_path(*, log_prefix="train-logs"):
    date = datetime.now()
    log_file_name = log_prefix + '_' + date.strftime("%d-%m-%Y")
    return os.path.join(common.LOG_PATH, log_file_name)
