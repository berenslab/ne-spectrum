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

from pathlib import Path


def plot_mnn(axs, df):

    lbl_txts = [
        ("mnist", "MNIST"),
        ("famnist", "Fashion MNIST"),
        ("kuzmnist", "Kuzushiji MNIST"),
        ("kannada", "Kannada MNIST"),
        ("treutlein", "Chimp organoid"),
        ("treutlein_409b2", "Human organoid"),
        ("tasic", "Mouse cortex"),
        ("hydra", "Hydra"),
        ("zfish", "Zebrafish"),
    ]

    umap = df.loc["umap"]
    fa2 = df.loc["fa2"]
    df = df.loc["tsne"]

    cmap = plt.get_cmap('tab10')
    for i, (ax, (lbl, title)) in enumerate(zip(axs.flat, lbl_txts)):
        ax.set_title(title, fontsize="medium")
        line, = ax.plot(df.index, df[lbl], color="black") # mpl.colors.rgb2hex(cmap(i))
        c = line.get_color()

        x, y = umap.index[0], umap[lbl]
        ax.scatter(x, y, c=c, s=15, zorder=3)
        ax.text(x + 0.5, y, "UMAP", horizontalalignment="left", verticalalignment="bottom")

        x, y = fa2.index[0], fa2[lbl]
        ax.scatter(x, y, c=c, s=15, zorder=3)
        ax.text(x, y * 1.15, "FA2", horizontalalignment="left", verticalalignment="bottom")

        jnb_msc.plot.despine(ax)

    for ax in axs[-1, :]:
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_xscale("log")
        ax.set_xlabel(r"Exaggeration ($\rho$)")

    for ax in axs[:, 0]:
        ax.set_ylabel("$k$NN recall ($k=15$)")
        ax.set_ylim(bottom=0)


if __name__ == "__main__":

    mnn_fpath = "../stats/neigh-presv/mutual-neigh-frac.csv"
    # redo.redo_ifchange(mnn_fpath)

    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple("../data/mnist/")
    rcfile = plotter.rc

    df = pd.read_csv(mnn_fpath,
                     index_col=["algo", "rho"],
                     comment="#")

    with plt.rc_context(fname=rcfile):
        fig, axs = plt.subplots(
            3,
            3,
            sharex=True,
            sharey=True,
            figsize=(5.5 * 0.6, 5.5 * 0.6),
            constrained_layout=True,
        )

        plot_mnn(axs, df)

        fig.savefig(sys.argv[3], format="pdf")
