import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import common


def model_file_path(model_id):
    date = datetime.now()
    ts = date.strftime("%d_%m_%Y-%H:%M:%S")
    model_fname = '-'.join((model_id, ts))
    return os.path.join(common.MODEL_PATH, model_fname)


def model_results_path(model_id):
    if not os.path.exists(common.OUT_DIR):
        os.mkdir(common.OUT_DIR)

    model_dir = os.path.join(common.OUT_DIR, model_id)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    date = datetime.now()
    ts = date.strftime("%d_%m_%Y-%H:%M:%S")
    base_dir_name = '-'.join((model_id, ts))
    base_dir = os.path.join(model_dir, base_dir_name)
    os.mkdir(base_dir)
    print("Created model directory:", base_dir)

    dir_paths = {}
    for dir_name, sub_dirs in common.DIR_STRUCTURE.items():
        result_dir = os.path.join(base_dir, dir_name)
        os.mkdir(result_dir)
        dir_paths[dir_name] = result_dir
        for sub_dir_name in sub_dirs:
            sub_dir = os.path.join(result_dir, sub_dir_name)
            os.mkdir(sub_dir)
            dir_paths[sub_dir_name] = sub_dir

    return base_dir, dir_paths


def train_results_path(base_dir, prefix, layer):
    results_fname = "{}-{}.npy".format(prefix, layer)
    return os.path.join(base_dir, common.METRICS_DIR, common.TRAINING_DIR, results_fname)


def prune_results_path(base_dir, prefix, layer):
    results_fname = "{}-{}.npy".format(prefix, layer)
    return os.path.join(base_dir, common.METRICS_DIR, common.PRUNING_DIR, results_fname)

"""
def prune_results_path(model_id, layer, pruning_method, pruning_ratio, metric):
    date = datetime.now()
    ts = date.strftime("%d_%m_%Y-%H:%M:%S")
    results_fname = '-'.join((model_id, layer, pruning_method, pruning_ratio, metric, ts))
    return os.path.join(common.PRUNE_METRICS_PATH, results_fname)
"""


def plot_test_acc(base_dir):
    acc_path = os.path.join(base_dir, common.METRICS_DIR, common.TRAINING_DIR, common.ACCURACY_FNAME)
    test_accuracies = np.load(acc_path)
    plt.plot(test_accuracies)
    plt.title('Testset accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()


def plot_train_fps(fp_paths, layers):
    n_plots = len(layers)
    fig, axs = plt.subplots(n_plots, 1, figsize=(8, 20), sharex=True)
    for i, layer in enumerate(layers):
        layer_fp = np.load(fp_paths[layer])
        axs[i].plot(layer_fp)
        axs[i].set_title(f"Frame potential of {layer}")

    fig.suptitle("Frame potential (FP) during training", fontsize=16)
    fig.text(0.5, 0.04, 'Epoch', ha='center')
    fig.text(0.04, 0.5, 'FP', va='center', rotation='vertical')


def draw_acc_plot(layer, exp_acc_fname, pruning_acc_fname, *, figsize=(25, 12), fig_name=""):
    plt.figure(figsize=figsize)

    pruning_acc = np.load(pruning_acc_fname, allow_pickle=True).item()
    for pruning_method, acc in pruning_acc.items():
        n_obs = len(acc)
        plt.plot(acc, label=pruning_method)

    if exp_acc_fname:
        exp_acc = np.load(exp_acc_fname)
        random_accuracies = exp_acc.mean(axis=0)
        random_std_devs = exp_acc.std(axis=0)
        plt.errorbar(range(n_obs), random_accuracies, random_std_devs, linestyle='None',
                     marker='.', label="random pruning")

    plt.title("Accuracy after pruning " + layer)
    elem_type = "neurons" if layer[:2] == "fc" else "filters"
    plt.xlabel("Pruned " + elem_type)
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="best")

    if fig_name:
        fig_path = os.path.join(common.FIGURE_PATH, fig_name)
        plt.savefig(fig_path, format="png")
    else:
        plt.show()


def draw_fps_plot(layer, exp_fps_fname, pruning_fps_fname, *, figsize=(25, 12), fig_name=""):
    plt.figure(figsize=figsize)

    pruning_fps = np.load(pruning_fps_fname, allow_pickle=True).item()
    for pruning_method, fps in pruning_fps.items():
        n_obs = len(fps)
        plt.plot(fps, label=pruning_method)

    if exp_fps_fname:
        exp_fps = np.load(exp_fps_fname)
        random_fps = exp_fps.mean(axis=0)
        random_std_devs = exp_fps.std(axis=0)
        plt.errorbar(range(n_obs), random_fps, random_std_devs, linestyle='None',
                     marker='.', label="random pruning")

    plt.title("Frame potential after pruning " + layer)
    elem_type = "neurons" if layer[:2] == "fc" else "filters"
    plt.xlabel("Pruned " + elem_type)
    plt.ylabel("FP")
    plt.legend(loc="best")

    if fig_name:
        fig_path = os.path.join(common.FIGURE_PATH, fig_name)
        plt.savefig(fig_path, format="png")
    else:
        plt.show()


def draw_sched_plot(layer, acc_fname, pruning_method, pruning_ratio, *, figsize=(14, 7), fig_name=""):
    plt.figure(figsize=figsize)

    accuracies = np.load(acc_fname, allow_pickle=True).item()
    for init_time, acc in accuracies.items():
        if init_time == -1:
            plt.plot(accuracies[init_time], label="No pruning")
        elif init_time == 0:
            plt.plot(accuracies[init_time], label="{}% random_pruning before training".format(int(pruning_ratio * 100)))
        else:
            plt.plot(accuracies[init_time],
                     label="{}% {} after iteration {}".format(int(pruning_ratio * 100), pruning_method, init_time))
    plt.title(f"Testset accuracy (pruning {layer})")
    plt.xlabel("Time")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="best")

    if fig_name:
        fig_path = os.path.join(common.FIGURE_PATH, fig_name)
        plt.savefig(fig_path, format="png")
    else:
        plt.show()
