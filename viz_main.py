import json
from flask import Flask, request, redirect, url_for, render_template, send_file

import common

from processing.snapshots import Snapshots


app = Flask(__name__)


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
            graph, epochs = s.training_graph(layer, merged=True)
        else:
            s = Snapshots(base_dir)
            graph, epochs = s.training_graph(layer)
        graph_data = json.dumps(graph, indent=4)
        n_nodes = s.get_weights(layer).shape[1]
        data = {"graph_data": graph_data, "max_epoch": epochs-1, "n_nodes": n_nodes}
        return render_template("merged.html", data=data)  # debug


def start(**kwargs):
    app.run(host="0.0.0.0", **kwargs)


if __name__ == '__main__':
    start(debug=True)
