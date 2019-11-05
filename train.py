import argparse
import torch.nn as nn
import torch.optim as optim

from processing.snapshots import Snapshots
from data import get_mnist, get_cifar
from experiment import Experiment


def save_results(output_dir, layers):
    s = Snapshots(output_dir)
    fp_paths = {}
    ip_paths = {}
    norms_paths = {}
    graph_specs = {}
    for layer in layers:
        fp_path, ip_path, norms_path = s.save_computed_metrics(layer)
        fp_paths[layer] = fp_path
        ip_paths[layer] = ip_path
        norms_paths[layer] = norms_path
        graph_specs[layer] = s.create_adjacency(layer)

    return fp_paths, ip_paths, norms_paths, graph_specs


def fit(train_batch_size, test_batch_size, epochs, model_class, criterion_class,
        optimizer_class, optim_kwargs, use_cifar):

    dataset = get_cifar if use_cifar else get_mnist
    trainloader, testloader, classes = dataset(train_batch_size, test_batch_size)

    model = eval(model_class)()
    criterion = eval(criterion_class)()
    optimizer = eval(optimizer_class)(model.parameters(), **optim_kwargs)
    layers = {name.split('.')[0] for name, _ in model.named_parameters()}
    layers = list(sorted(layers))

    e = Experiment(model, criterion, optimizer, optim_kwargs)
    test_accuracies, base_dir = e.fit(trainloader, testloader, epochs, save_results=True)
    fp_paths, ip_paths, norms_paths, graph_specs = save_results(base_dir, layers)


if __name__=="__main__":
    # Defines all parser arguments when launching the script directly in terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("train_batch_size", type=int, help="training batch size")
    parser.add_argument("test_batch_size", type=int, help="testing batch size")
    parser.add_argument("epochs", type=int, help="number of training epochs")
    parser.add_argument("model_class", type=str, help="model class")
    parser.add_argument("criterion_class", type=str, help="criterion class")
    parser.add_argument("optimizer_class", type=str, help="optimizer class")
    parser.add_argument("optim_args", nargs='*')
    parser.add_argument("-cif", "--cifar", help="use cifar dataset", action="store_true")
    args = parser.parse_args()

    exec(f"from architecture.models import {args.model_class}")

    args_dict = {}
    for t in args.optim_args:
        k, v = t.split('=')
        args_dict[k] = float(v)

    fit(args.train_batch_size, args.test_batch_size, args.epochs, args.model_class,
        args.criterion_class, args.optimizer_class, args_dict, args.cifar)
