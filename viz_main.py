import json
from flask import Flask, jsonify, request, redirect, render_template, url_for

import common

from processing.snapshots import Snapshots


app = Flask(__name__)

graph_data = None


@app.route('/', methods=["GET", "POST"])
def root():
    """ Flask method of the root page with the forms """
    with open(common.MODEL_SPECS_PATH, 'r') as fp:
        models = json.load(fp)

    if request.method == "GET":
        return render_template("form.html", model_list=list(models.keys()))
    elif request.method == "POST":
        if request.form.get("btn") == "Select model":
            base_dir = request.form.get("model")
            return redirect(url_for('params', base_dir=base_dir))


@app.route('/params', methods=["GET", "POST"])
def params():
    """ Flask method of the root page with the forms """
    with open(common.MODEL_SPECS_PATH, 'r') as fp:
        models = json.load(fp)

    base_dir = request.args.get("base_dir")
    if request.method == "GET":
        layers = models[base_dir]
        layers.sort()
        return render_template("params.html", model_list=list(models.keys()), layers=layers)
    elif request.method == "POST":
        if request.form.get("btn") == "Training graph":
            layer = request.form.get("layer")
            var_base_dir = request.form.get("model_variant", "")

            return redirect(url_for('result', base_dir=base_dir, layer=layer, var_base_dir=var_base_dir))


@app.route("/result", methods=["GET"])
def result():
    """ Flask method of the training results page """
    if request.method == "GET":
        base_dir = request.args.get("base_dir")
        layer = request.args.get("layer")
        var_base_dir = request.args.get("var_base_dir")

        # Returns the HTML file and renders the locally saved file
        if var_base_dir:
            s = Snapshots(var_base_dir)
            adjacency, kernel_width = s.create_adjacency(layer, merged=True)
            graph, epochs = s.training_graph(layer, adjacency, merged=True)
        else:
            s = Snapshots(base_dir)
            adjacency, kernel_width = s.create_adjacency(layer)
            graph, epochs = s.training_graph(layer, adjacency)

        global graph_data
        graph_data = graph

        n_nodes = s.get_weights(layer).shape[1]

        max_connect = max(map(lambda l: l["value"], graph["links"]))
        all_norms = []
        for node in graph["nodes"]:
            all_norms += node["norm"].values()
        max_norm = max(all_norms)

        data = {"max_epoch": epochs - 1, "n_nodes": n_nodes, "max_connect": max_connect,
                "max_norm": max_norm, "kernel_width": kernel_width}
        return render_template("merged.html", data=data)


@app.route("/get-graph", methods=["GET"])
def get_graph():
    return jsonify(graph_data)


def start(**kwargs):
    app.run(host="0.0.0.0", **kwargs)


if __name__ == '__main__':
    start(debug=True)
