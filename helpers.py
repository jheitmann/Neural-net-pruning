import os
import matplotlib.pyplot as plt
import numpy as np
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
