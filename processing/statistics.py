import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

import common


def weight_dist(inner_products):
    # Transform inner-products to distances between 0 and 1
    ips_mod = -0.5 * (inner_products - 1)
    kernel_width = ips_mod.mean()
    adjacency = np.exp(-ips_mod ** 2 / (kernel_width ** 2))
    thresh = np.exp(-0.5 ** 2 / (kernel_width ** 2))

    # No self-loops
    for time in range(inner_products.shape[0]):
        for node_idx in range(inner_products.shape[1]):
            adjacency[time, node_idx, node_idx] = 0.

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
    return adjacency, kernel_width


def plot_connected_components(adjacency, thresholds):
    cc_numbers = []
    for cut_off in thresholds:
        w = adjacency.copy()
        w[w < cut_off] = 0.
        n_models = w.shape[0]
        G = nx.from_numpy_matrix(w[n_models - 1])
        n_connected = nx.number_connected_components(G)
        cc_numbers.append(n_connected)

    fig = plt.figure(figsize=(12, 7))
    plt.scatter(thresholds, cc_numbers, s=10)
    plt.title("Number of connected components in graph with weights below cut-off removed")
    plt.xlabel("Cut-off value")
    plt.ylabel("Number of connected components")
    plt.show()


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


def training_graph(adjacency, weight_norms, *, graph_name=""):
    graph = {}
    n_epochs = adjacency.shape[0]
    n_nodes = adjacency.shape[1]

    graph["nodes"] = []
    for node_id in range(n_nodes):
        norms = {str(epoch): float("{:.3f}".format(weight_norms[epoch, node_id])) for epoch in range(n_epochs)}
        node_entry = {"id": str(node_id), "fraction": norms}
        graph["nodes"].append(node_entry)

    graph["links"] = []
    for epoch in range(n_epochs):
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edge_weight = adjacency[epoch, i, j]
                if edge_weight:
                    edge_entry = {"source": str(i), "target": str(j), "value": float("{:.3f}".format(edge_weight)),
                                  "year": epoch}
                    graph["links"].append(edge_entry)

    if graph_name:
        graph_fname = os.path.join(common.VIZ_DIR, graph_name)
        with open(graph_fname, 'w') as fp:
            json.dump(graph, fp, indent=4)

    return graph
