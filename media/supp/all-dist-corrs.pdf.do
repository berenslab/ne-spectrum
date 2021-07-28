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


def plot_dcor(fig, axs, df):
    datasets = [
        ("mnist",            "MNIST"),
        ("famnist",          "Fashion MNIST"),
        ("kuzmnist",         "Kuzushiji MNIST"),
        ("kannada",          "Kannada MNIST"),
        ("treutlein",        "Chimp organoid"),
        ("treutlein_409b2",  "Human organoid"),
        ("tasic",            "Mouse cortex"),
        ("hydra",            "Hydra"),
        ("zfish",            "Zebrafish"),
        # "treutlein_h9",
        # "gauss_devel",
    ]
    df_full = df
    rhos = df.loc["umap"].index

    fd = jnb_msc.plot.ScatterMultiple.get_letterdict()

    for algo, letter in zip(["umap", "fa2"], "ab"):
        title = algo.upper()
        df = df_full.loc[algo]
        ax = axs[algo]

        for key, label in datasets:
            zorder = 3 if key == "mnist" else None
            line, = ax.plot(rhos, df[key], label=label, zorder=zorder)
            c = line.get_color()
            ix = df[key].argmax()
            x = rhos[ix]
            y = df[key].iloc[ix]
            clip_on = False # y < 0.9
            ax.scatter([x], [y], s=30, zorder=zorder, clip_on=clip_on)

        jnb_msc.plot.despine(ax)
        ax.set_xscale("log")
        ax.set_xlabel(r"Exaggeration ($\rho$)")
        ax.set_xlim(rhos[0], rhos[-1])
        ax.set_ylim(top=1, bottom=0.9)

        ax.set_title(title)
        ax.set_title(letter, fontdict=fd, loc="left")

    _, _, ymin, ymax = axs["umap"].axis()
    axs["umap"].plot([4, 4], [ymin, ymax], c="xkcd:light grey", alpha=0.5, zorder=1)
    axs["fa2"].plot([30, 30], [ymin, ymax], c="xkcd:light grey", alpha=0.5, zorder=1)

    axs["fa2"].tick_params(labelleft=False)
    ax = axs["umap"]
    ax.set_ylabel("Distance correlation")

    handles, labels = ax.get_legend_handles_labels()
    axs["legend"].set_axis_off()
    axs['legend'].legend(
        handles, labels, loc="upper center", ncol=5, mode="expand", frameon=False,
    )





if __name__ == "__main__":

    mnn_fpath = "../../stats/dist-corr/dist-corr.csv"
    redo.redo_ifchange(mnn_fpath)

    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple("../../data/mnist/")
    rcfile = plotter.rc

    df = pd.read_csv(mnn_fpath,
                     index_col=["algo", "rho"],
                     comment="#")

    with plt.rc_context(fname=rcfile):
        fig, axs = plt.subplot_mosaic(
            [['umap', 'fa2'],
             ['legend', 'legend']],
            # sharey=True,
            gridspec_kw={
                'height_ratios': [1, 0.001]
            },
            figsize=(5.5, 2.5),
            constrained_layout=True,
        )
        axs["umap"].sharey(axs["fa2"])


        plot_dcor(fig, axs, df)

        fig.savefig(sys.argv[3], format="pdf")
