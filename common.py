import os


OUT_DIR = "out"
FIGURE_PATH = os.path.join(OUT_DIR, "figures")
MODEL_PATH = os.path.join(OUT_DIR, "models")
METRICS_PATH = os.path.join(OUT_DIR, "metrics")
TRAIN_METRICS_PATH = os.path.join(METRICS_PATH, "training")
PRUNE_METRICS_PATH = os.path.join(METRICS_PATH, "pruning")

SEED = 2019
