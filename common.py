import os


OUT_DIR = "out"
VIZ_DIR = "viz"
METRICS_DIR = "metrics"
SNAPSHOTS_DIR = "snapshots"
FIGURES_DIR = "figures"
TRAINING_DIR = "training"
PRUNING_DIR = "pruning"

MODEL_PATH = os.path.join(OUT_DIR, "models")
METRICS_PATH = os.path.join(OUT_DIR, "metrics")
PRUNE_METRICS_PATH = os.path.join(METRICS_PATH, "pruning")

MODEL_SPECS_PATH = os.path.join(OUT_DIR, "model_specs.json")
ACCURACY_FNAME = "accuracies.npy"
FP_PREFIX = "frame_potentials"
NORM_PREFIX = "weight_norms"
IP_PREFIX = "inner_products"
ADJACENCY_PREFIX = "adjacency"

DIR_STRUCTURE = {
    METRICS_DIR: [TRAINING_DIR, PRUNING_DIR],
    SNAPSHOTS_DIR: []
}

SEED = 2019
