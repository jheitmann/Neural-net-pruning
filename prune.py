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
    for pruning_idx in fp_pruning(model, layer, pruning_iters, increase_fp=True):
        yield pruning_idx


def min_fp_pruning(model, layer, pruning_iters):
    for pruning_idx in fp_pruning(model, layer, pruning_iters, increase_fp=False):
        yield pruning_idx


def max_mag_pruning(model, layer, pruning_iters):
    squared_norms = model.compute_squared_norms(layer)
    squared_norms.sort()
    for i in range(pruning_iters):
        _, pruning_idx = squared_norms[i]
        yield pruning_idx


def min_mag_pruning(model, layer, pruning_iters):
    squared_norms = model.compute_squared_norms(layer)
    squared_norms.sort(reverse=True)
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


def prune_and_test(experiment, testloader, layer, pruning, pruning_ratio, *, save_results=False, log_interval=10):
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

        if (pruning_iter+1) % log_interval == 0:
            print("Pruning round: [{:3d}/{:3d} ({:.0f}%)]".format(
                pruning_iter+1, pruning_iters, 100. * (pruning_iter+1) / pruning_iters))

    if save_results:
        save_pruning_meta(experiment.model, layer, pruning_iters, pruning.__name__, test_accuracies, frame_potentials)
    
    return test_accuracies, frame_potentials


def random_pruning_rounds(experiment, testloader, layer, n_rounds, pruning_ratio, *, save_results=False):
    exp_acc = []
    exp_fps = []
    for i in range(n_rounds):
        print("Random pruning experiment N°", i+1)
        experiment.init_model()
        test_accuracies, frame_potentials = prune_and_test(experiment, testloader, layer, random_pruning, pruning_ratio)
        exp_acc.append(test_accuracies)
        exp_fps.append(frame_potentials)
    
    if save_results:
        save_pruning_meta(experiment.model, layer, len(exp_acc), random_pruning.__name__, exp_acc, exp_fps)
            
    return exp_acc, exp_fps

"""
def fit_and_prune(experiment, epochs, criterion, optimizer, layer, pruning, pruning_ratio, 
                    pruning_time, stop_time, *, save_results=False, log_interval=100):
    initial_acc, initial_fps = experiment.test(criterion, monitored=[layer])
    test_accuracies = [initial_acc]  # test before step
    frame_potentials = [initial_fps[layer]]
    time = 1
    for epoch in range(1, epochs + 1):
        for accuracy, layer_fp in experiment.custom_train(criterion, optimizer, epoch, [layer], log_interval): 
            test_accuracies.append(accuracy)
            frame_potentials.append(layer_fp[layer])
            if time == pruning_time:
                acc, fp = prune_and_test(experiment, criterion, layer, pruning, pruning_ratio)
                # does something need to be done?
            elif time == stop_time:
                return test_accuracies, frame_potentials
            time += 1

    #if save_results:
    #    save_training_meta(experiment.model, epochs, test_accuracies, frame_potentials)
    
    return test_accuracies, frame_potentials
"""

def pruning_schedule(experiment, trainloader, testloader, epochs, 
                    test_interval, saving_times, layer, pruning_ratio):
    trainable = {0: experiment}
    initial_acc, _ = experiment.test(testloader, [])
    accuracies = {0: [initial_acc]}
    n_elem = getattr(experiment.model, layer).weight.shape[0]
    pruning_iters = int(pruning_ratio * n_elem)

    time = 1
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            
            for e in trainable.values():
                training_loss = e.batch_train(data, target)
            
            if batch_idx % test_interval == (test_interval-1):
                print("Evaluating model accuracies [{:3d}/{:3d}]".format(time, epochs * (len(trainloader) // test_interval)))
                if time in saving_times:
                    new_experiment = experiment.clone()
                    new_experiment.model.prune_layer(layer, max_fp_pruning, pruning_iters)
                    trainable[time] = new_experiment
                    accuracies[time] = accuracies[0].copy()

                for init_time, e in trainable.items():
                    test_accuracy, _ = e.test(testloader, [])
                    accuracies[init_time].append(test_accuracy)
                
                time += 1

    return trainable, accuracies
