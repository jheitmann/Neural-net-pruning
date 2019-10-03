import numpy as np

import common
import helpers

np.random.seed(common.SEED)


def save_pruning_meta(model, layer, pruning_iters, pruning_method, test_accuracies, frame_potentials):
    acc_fname = helpers.prune_results_path(model.model_ID(), layer, pruning_iters, pruning_method, "acc")
    np.save(acc_fname, test_accuracies)
    print("Saved validation accuracies to:", acc_fname + ".npy")
    fp_fname = helpers.prune_results_path(model.model_ID(), layer, pruning_iters, pruning_method, "fp")
    np.save(fp_fname, frame_potentials)
    print("Saved frame potentials to:", fp_fname + ".npy")


def fp_pruning(model, layer, pruning_iters, increase_fp):
    for i in range(pruning_iters):
        partial_fps = model.selective_fps(layer)
        _, min_idx = min(partial_fps)
        _, max_idx = max(partial_fps)
        pruning_idx = max_idx if increase_fp else min_idx
        yield pruning_idx


def max_fp_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(layer, pruning_iters, increase_fp=True):
        yield pruning_idx


def min_fp_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(layer, pruning_iters, increase_fp=False):
        yield pruning_idx


def magnitude_pruning(model, layer, pruning_iters):
    squared_norms = model.compute_squared_norms(layer)
    squared_norms.sort()
    for i in range(pruning_iters):
        _, pruning_idx = squared_norms[i]
        yield pruning_idx


def random_pruning(model, layer, pruning_iters):
    param = getattr(model, layer)
    unpruned_indices = param.unpruned_parameters()
    np.random.shuffle(unpruned_indices)
    for i in range(pruning_iters):
        pruning_idx = unpruned_indices[i]
        yield pruning_idx


def prune_and_test(experiment, criterion, layer, pruning, pruning_ratio, *, save_results=False, log_interval=10):
    initial_acc, initial_fps = experiment.test(criterion, monitored=[layer])
    test_accuracies = [initial_acc]
    frame_potentials = [initial_fps[layer]]

    n_elem = getattr(experiment.model, layer).weight.shape[0]
    pruning_iters = int(pruning_ratio * n_elem)
    
    for pruning_iter, pruning_idx in enumerate(pruning(experiment.model, layer, pruning_iters)):
        experiment.model.prune_element(layer, pruning_idx)
        accuracy, layer_fp = experiment.test(criterion, monitored=[layer])
        test_accuracies.append(accuracy)
        frame_potentials.append(layer_fp[layer])

        if (pruning_iter+1) % log_interval == 0:
            print("Pruning round: [{:3d}/{:3d} ({:.0f}%)]".format(
                pruning_iter+1, pruning_iters, 100. * (pruning_iter+1) / pruning_iters))

    if save_results:
        save_pruning_meta(experiment.model, layer, pruning_iters, pruning.__name__, test_accuracies, frame_potentials)
    
    return test_accuracies, frame_potentials


def random_pruning_rounds(experiment, criterion, layer, n_rounds, pruning_ratio, *, save_results=False):
    exp_acc = []
    exp_fps = []  
    for i in range(n_rounds):
        print("Random pruning experiment N°", i+1)
        experiment.init_model()
        test_accuracies, frame_potentials = prune_and_test(experiment, criterion, layer, random_pruning, pruning_ratio)
        exp_acc.append(test_accuracies)
        exp_fps.append(frame_potentials)
    
    if save_results:
        save_pruning_meta(experiment.model, layer, len(exp_acc), "rnd", exp_acc, exp_fps)
            
    return exp_acc, exp_fps
    