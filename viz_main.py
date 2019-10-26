import pickle
from flask import Flask, request, redirect, url_for, render_template, send_file

import common

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def root():
    """ Flask method of the root page with the forms """
    with open(common.MODEL_LIST_PATH, 'rb') as fp:
        models = pickle.load(fp)

    if request.method == "GET":
        return render_template("form.html", model_list=models)
    elif request.method == "POST":
        if request.form.get("btn") == "Compute route":
            stn_from, stn_to, qvalue, date, time = (request.form.get(k) for k in ("from", "to", "qvalue", "date", "time"))
            return redirect(url_for('result', stn_from=stn_from, stn_to=stn_to, qvalue=qvalue, date=date, time=time))
        elif request.form.get("btn") == "Compute isochrones":
            stn_origin = request.form.get("origin")
            return redirect(url_for('iso', stn_origin=stn_origin))


@app.route("/result", methods=["GET"])
def result():
    """ Flask method of the route planning results page """
    if request.method == "GET":
        tolerance = int(request.args.get("qvalue"))*.01

        # Returns the HTML file and renders the localy saved file
        steps = None
        return render_template("result.html", steps=steps)


def start(**kwargs):
    app.run(host="0.0.0.0", **kwargs)


if __name__ == '__main__':
    start(debug=True)