import numpy as np
from copy import deepcopy

import common
import helpers

np.random.seed(common.SEED)


def save_pruning_meta(model, layer, pruning_method, pruning_ratio, test_accuracies, frame_potentials):
    acc_fname = helpers.prune_results_path(model.model_id(), layer, pruning_method, pruning_ratio, "acc")
    np.save(acc_fname, test_accuracies)
    print("Saved validation accuracies to:", acc_fname + ".npy")
    fp_fname = helpers.prune_results_path(model.model_id(), layer, pruning_method, pruning_ratio, "fp")
    np.save(fp_fname, frame_potentials)
    print("Saved frame potentials to:", fp_fname + ".npy")


def fp_pruning(model, layer, pruning_iters, *, least_decrease=True, l2_norm=True, normalize=True):
    for i in range(pruning_iters):
        partial_fps = model.selective_correlation(layer, l2_norm, normalize)
        _, min_idx = min(partial_fps)
        _, max_idx = max(partial_fps)
        pruning_idx = max_idx if least_decrease else min_idx
        yield pruning_idx


def max_fp_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(model, layer, pruning_iters):
        yield pruning_idx


def min_fp_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(model, layer, pruning_iters, least_decrease=False):
        yield pruning_idx


def max_ip_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(model, layer, pruning_iters, l2_norm=False):
        yield pruning_idx


def min_ip_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(model, layer, pruning_iters, least_decrease=False, l2_norm=False):
        yield pruning_idx


def max_fp_mod_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(model, layer, pruning_iters, normalize=False):
        yield pruning_idx


def min_fp_mod_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(model, layer, pruning_iters, least_decrease=False, normalize=False):
        yield pruning_idx


def max_mag_pruning(model, layer, pruning_iters):  # can't prune in cycles
    weight_norms = model.compute_norms(layer)
    _, indices = weight_norms.sort()
    for idx in indices.tolist()[:pruning_iters]:
        yield idx


def min_mag_pruning(model, layer, pruning_iters):  # can't prune in cycles
    weight_norms = model.compute_norms(layer)
    _, indices = weight_norms.sort(descending=True)
    for idx in indices.tolist()[:pruning_iters]:
        yield idx


def random_pruning(model, layer, pruning_iters):
    param = getattr(model, layer)
    unpruned_indices = param.unpruned_parameters()
    np.random.shuffle(unpruned_indices)
    for i in range(pruning_iters):
        pruning_idx = unpruned_indices[i]
        yield pruning_idx


def prune_layer(model, layer, pruning, pruning_ratio):
    n_elem = len(getattr(model, layer).unpruned_parameters())
    pruning_iters = int(pruning_ratio * n_elem)
    for pruning_iter, pruning_idx in enumerate(pruning(model, layer, pruning_iters)):
        model.prune_element(layer, pruning_idx)


def prune_and_test(experiment, testloader, layer, pruning, pruning_ratio, *, save_results=False, log_interval=10):  # changeme for ips
    initial_acc, initial_fps = experiment.test(testloader, [layer])
    test_accuracies = [initial_acc]
    frame_potentials = [initial_fps[layer]]

    n_elem = getattr(experiment.model, layer).weight.shape[0]
    pruning_iters = int(pruning_ratio * n_elem)

    for pruning_iter, pruning_idx in enumerate(pruning(experiment.model, layer, pruning_iters)):
        experiment.model.prune_element(layer, pruning_idx)
        accuracy, layer_fp = experiment.test(testloader, [layer])
        test_accuracies.append(accuracy)
        frame_potentials.append(layer_fp[layer])

        if (pruning_iter + 1) % log_interval == 0:
            print("Pruning round: [{:3d}/{:3d} ({:.0f}%)]".format(
                pruning_iter + 1, pruning_iters, 100. * (pruning_iter + 1) / pruning_iters))

    if save_results:
        save_pruning_meta(experiment.model, layer, pruning.__name__, pruning_ratio, test_accuracies, frame_potentials)

    return test_accuracies, frame_potentials


def random_pruning_rounds(experiment, testloader, layer, n_rounds, pruning_ratio, *, save_results=False):
    exp_acc = []
    exp_fps = []
    for i in range(n_rounds):
        print("Random pruning experiment N°", i + 1)
        experiment.init_model()
        test_accuracies, frame_potentials = prune_and_test(experiment, testloader, layer, random_pruning, pruning_ratio)
        exp_acc.append(test_accuracies)
        exp_fps.append(frame_potentials)

    if save_results:
        save_pruning_meta(experiment.model, layer, random_pruning.__name__, pruning_ratio, exp_acc, exp_fps)

    return exp_acc, exp_fps


def evaluate_models(testloader, trainable, layers, accuracies, frame_potentials):  # changeme for ips
    for init_time, e in trainable.items():
        accuracy, layer_fp = e.test(testloader, layers)
        accuracies.setdefault(init_time, []).append(accuracy)
        model_fps = frame_potentials.setdefault(init_time, {})
        for layer, fp in layer_fp.items():
            model_fps.setdefault(layer, []).append(fp)


def pruning_schedule(experiment, trainloader, testloader, epochs, test_interval,
                     saving_times, layers, pruning, pruning_ratio, *, save_results=False):  # changeme for ips
    initial_pruning = experiment.clone()
    for layer in layers:
        prune_layer(initial_pruning.model, layer, random_pruning, pruning_ratio)
    trainable = {-1: experiment, 0: initial_pruning}
    accuracies, frame_potentials = {}, {}
    evaluate_models(testloader, trainable, layers, accuracies, frame_potentials)

    max_t = epochs * (len(trainloader) // test_interval)
    time = 1
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):

            for e in trainable.values():
                training_loss = e.batch_train(data, target)

            if batch_idx % test_interval == (test_interval - 1):
                print("Evaluating model accuracies [{:3d}/{:3d}]".format(time, max_t))
                if time in saving_times:
                    new_experiment = experiment.clone()
                    for layer in layers:
                        prune_layer(new_experiment.model, layer, pruning, pruning_ratio)
                    trainable[time] = new_experiment
                    accuracies[time] = accuracies[-1].copy()
                    frame_potentials[time] = deepcopy(frame_potentials[-1])

                evaluate_models(testloader, trainable, layers, accuracies, frame_potentials)
                time += 1

    if save_results:
        save_pruning_meta(experiment.model, '-'.join(layers), "scheduled-pruning",
                          pruning_ratio, accuracies, frame_potentials)

    return accuracies, frame_potentials


def iterative_pruning(experiment, trainloader, testloader, epochs, test_interval,
                      pruning, pruning_args, *, save_results=False):
    pruning_start = pruning_args["start"]  # dict with layer keys
    pruning_end = pruning_args["end"]  # dict with layer keys
    inter_pruning_iters = pruning_args["inter_pruning"]  # dict with layer keys
    pruning_ratio = pruning_args["ratio"]

    iterative_ratio = {}
    for layer, start in pruning_start.items():
        end = pruning_end[layer]
        r = inter_pruning_iters[layer]
        ratio = pruning_ratio[layer]
        k = 1 + (end - start) // r
        iterative_ratio[layer] = 1 - (1-ratio)**(1 / k)

    accuracies = []
    max_t = 1 + epochs * (len(trainloader) // test_interval)  # probably ceil
    time = 0

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            if batch_idx % test_interval == 0:  # (test_interval - 1)
                for layer, start in pruning_start.items():
                    end = pruning_end[layer]
                    r = inter_pruning_iters[layer]
                    if start <= time <= end and (time-start) % r == 0:
                        print("Pruning layer", layer)
                        ratio = iterative_ratio[layer]
                        prune_layer(experiment.model, layer, pruning, ratio)
                # Test
                print("Evaluating model accuracy [{:3d}/{:3d}]".format(time, max_t))
                accuracy, _ = experiment.test(testloader, [])
                accuracies.append(accuracy)
                time += 1
            # Train
            experiment.batch_train(data, target)

    return accuracies
