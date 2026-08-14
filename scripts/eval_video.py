#!/usr/bin/env python3

import os
import pandas
import echonet
import sys
import sklearn
import math
import matplotlib.pyplot as plt
import collections
import numpy as np


def main():
    x = pandas.read_csv(sys.argv[1])
    y = x["Measurement"].values
    yhat = x["Prediction"].values

    if len(sys.argv) >= 3 and sys.argv[2] == "--average":
        y = collections.defaultdict(list)
        yhat = collections.defaultdict(list)
        for (fn, pred, meas) in x[["Filename", "Prediction", "Measurement"]].values:
            study = fn.split("_")[0]
            y[study].append(meas)
            yhat[study].append(pred)
        assert sorted(y) == sorted(yhat)
        studies = sorted(y)
        y = np.array([sum(y[s]) / len(y[s]) for s in studies])
        yhat = np.array([sum(yhat[s]) / len(yhat[s]) for s in studies])

    print("R2:   {:.3f} ({:.3f} - {:.3f})".format(*echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
    print("MAE:  {:.3f} ({:.3f} - {:.3f})".format(*echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
    print("ICC:  {:.3f} ({:.3f} - {:.3f})".format(*echonet.utils.bootstrap(y, yhat, icc)))
    print("RMSE: {:.3f} ({:.3f} - {:.3f})".format(*tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
    for thresh in [35, 50]:
        print("AUROC (>={:d}%):  {:.3f} ({:.3f} - {:.3f})".format(thresh, *echonet.utils.bootstrap(y >= thresh, yhat, sklearn.metrics.roc_auc_score)))
        fpr, tpr, _ = sklearn.metrics.roc_curve(y >= thresh, yhat)

        # Plot AUROC
        fig = plt.figure(figsize=(3, 3))
        plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
        plt.plot(fpr, tpr)

        plt.axis([-0.01, 1.01, -0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(sys.argv[1]), "roc_{}.svg".format(thresh)))
        plt.savefig(os.path.join(os.path.dirname(sys.argv[1]), "roc_{}.pdf".format(thresh)))
        plt.close(fig)

    # Plot actual and predicted EF
    fig = plt.figure(figsize=(3, 3))
    lower = min(y.min(), yhat.min())
    upper = max(y.max(), yhat.max())
    plt.scatter(y, yhat, color="k", s=1, edgecolor=None, zorder=2)
    plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
    plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("Actual EF (%)")
    plt.ylabel("Predicted EF (%)")
    plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
    plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(sys.argv[1]), "scatter.svg"))
    plt.savefig(os.path.join(os.path.dirname(sys.argv[1]), "scatter.pdf"))
    plt.close(fig)


def icc(y, yhat):
    n = len(y)
    ybar = (y.mean() + yhat.mean()) / 2
    s2 = (((y - ybar) ** 2).mean() + ((yhat - ybar) ** 2).mean()) / 2
    return ((y - ybar) * (yhat - ybar)).mean() / s2


if __name__ == "__main__":
    main()
