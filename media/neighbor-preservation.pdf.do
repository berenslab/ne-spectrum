#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect
import dcor
import scipy
import subprocess

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from pathlib import Path
from itertools import cycle
from matplotlib.legend_handler import HandlerTuple


def plot_mnn(axs, df):
    ax, leg_ax = axs

    umap = df.loc["umap"]
    fa2 = df.loc["fa2"]
    df = df.loc["tsne"]

    line, = ax.plot(df.index, df["mnist"],           label="MNIST", zorder=3)
    c = line.get_color()
    ax.plot(df.index, df["famnist"],         label="Fashion MNIST")
    ax.plot(df.index, df["kuzmnist"],        label="Kuzushiji MNIST")
    ax.plot(df.index, df["kannada"],         label="Kannada MNIST")
    ax.plot(df.index, df["treutlein"],       label="Chimp organoid")
    ax.plot(df.index, df["treutlein_409b2"], label="Human organoid")
    ax.plot(df.index, df["tasic"],           label="Mouse cortex")
    ax.plot(df.index, df["hydra"],           label="Hydra")
    ax.plot(df.index, df["zfish"],           label="Zebrafish")

    x, y = umap.index[0], umap["mnist"]
    ax.scatter(x, y, c=c, s=15, zorder=3)
    ax.text(x - 0.5, y, "UMAP", horizontalalignment="right", verticalalignment="top")

    x, y = fa2.index[0], fa2["mnist"]
    ax.scatter(x, y, c=c, s=15, zorder=3)
    ax.text(x, y - y * 0.1, "FA2", horizontalalignment="right", verticalalignment="top")

    jnb_msc.plot.despine(ax)
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_xscale("log")
    ax.set_xlabel(r"Exaggeration ($\rho$)")

    ax.set_ylabel("$k$NN recall ($k=15$)")
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    leg_ax.set_axis_off()
    leg_ax.legend(
        handles, labels, loc="center", ncol=1, mode=None, frameon=False
    )

    return ax


if __name__ == "__main__":

    mnn_fpath = "../stats/neigh-presv/mutual-neigh-frac.csv"
    redo.redo_ifchange(mnn_fpath)

    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple("../data/mnist/")
    rcfile = plotter.rc

    df = pd.read_csv(mnn_fpath,
                     index_col=["algo", "rho"],
                     comment="#")

    with plt.rc_context(fname=rcfile):
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(5.5 * 0.75, 1.75),
            constrained_layout=True,
            # gridspec_kw={
            #     "width_ratios": [1, 0.001],
            # }
        )

        plot_mnn(axs, df)

        fig.savefig(sys.argv[3], format="pdf")
