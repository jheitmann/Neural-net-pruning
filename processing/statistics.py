import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

import common


def weight_dist(adjacency, kernel_width):
    thresh = np.exp(-0.5**2 / (kernel_width**2))

    # Plot weight distribution of generated adjacency matrix
    data = adjacency.flatten()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20, 7))
    ax1.hist(data, bins=50)
    ax1.axvline(x=thresh, color='r', linestyle='dashed', linewidth=1.4, label="Orthogonal weight vectors")
    ax1.set_title('Distribution of entries of adjacency matrix')
    ax1.set_xlabel('Entry value')
    ax1.set_ylabel('Number of occurrences')
    ax1.legend()

    ax2.hist(data, bins=50, density=True, histtype='step', cumulative=-1)
    ax2.axhline(y=0.05, color='g', linewidth=1.4, label="5% of entries")
    x_minor_ticks = np.linspace(0., 0.975, 40)
    y_minor_ticks = np.linspace(0.05, 1, 20)
    ax2.set_xticks(x_minor_ticks, minor=True)
    ax2.set_yticks(y_minor_ticks, minor=True)
    ax2.grid(which='both')
    ax2.set_title("Reversed CDF of approximated distribution")
    ax2.set_xlabel('Entry value')
    ax2.set_ylabel('Reverse CDF')
    ax2.legend()

    plt.show()


def plot_connected_components(adjacency, min_cut_off, *, step=0.005, n_obs=55):
    max_cut_off = min_cut_off + (n_obs-1)*step
    cut_offs = np.linspace(min_cut_off, max_cut_off, n_obs)

    cc_numbers = []
    for cut_off in cut_offs:
        w = adjacency.copy()
        w[w < cut_off] = 0.
        n_models = w.shape[0]
        G = nx.from_numpy_matrix(w[n_models - 1])
        n_connected = nx.number_connected_components(G)
        cc_numbers.append(n_connected)

    fig = plt.figure(figsize=(12, 7))
    plt.scatter(cut_offs, cc_numbers, s=10)
    plt.title("Number of connected components in graph with weights below cut-off removed")
    plt.xlabel("Cut-off value")
    plt.ylabel("Number of connected components")
    plt.show()

    return cut_offs


def cc_max_norm(adjacency, weight_norms, thresholds):
    n_models = adjacency.shape[0]
    last_model = adjacency[n_models - 1]
    last_model_norms = weight_norms[n_models - 1]

    plot_data = []

    for cut_off in thresholds:
        w = last_model.copy()
        w[w < cut_off] = 0

        # Create a graph from sparse adjacency matrix
        G = nx.from_numpy_matrix(w)
        cc_ls = list(nx.connected_components(G))

        sizes = []
        magnitudes = []
        for connected_component in cc_ls:
            cc_size = len(connected_component)
            node_selection = list(connected_component)
            cc_max_mag = last_model_norms[node_selection].max()
            sizes.append(cc_size)
            magnitudes.append(cc_max_mag)

        plot_data.append((sizes, magnitudes))

    return plot_data


def cc_specs_plot(plot_data, cut_offs, idx):
    fig = plt.figure(figsize=(12, 7))
    sizes, magnitudes = plot_data[idx]
    plt.scatter(sizes, magnitudes)
    plt.title("Size vs max magnitude for cut-off {:.3f} ({} connected components)".format(cut_offs[idx], len(sizes)))
    plt.xlabel("Connected component size")
    plt.ylabel("Max magnitude of node in connected component")
