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

    leg_ax = axs["legend"]

    fd = jnb_msc.plot.ScatterMultiple.get_letterdict()

    for ax, s, letter in zip([axs["n/rho*Z"], axs["Z/n"]], ["", " Z/n"], "ab"):
        for key, label in datasets:
            zorder = 3 if key == "mnist" else None
            line, = ax.plot(df.index, df[key + s], label=label, zorder=zorder)

        jnb_msc.plot.despine(ax)
        ax.set_xscale("log")
        ax.set_xlabel(r"Exaggeration ($\rho$)")
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_yscale("log")
        # ax.set_ylim(top=1, bottom=0.9)

        ax.set_title(letter, fontdict=fd, loc="left")

    axs["n/rho*Z"].set_ylabel(r"$n/\rho Z$")
    axs["Z/n"].set_ylabel(r"$Z/n$")
    axs["Z/n"].set_ylim(10**1, 10**5)
    # ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    leg_ax.set_axis_off()
    leg_ax.legend(
        handles, labels, loc="upper center", ncol=5, mode="expand", frameon=False,
    )

    return ax


if __name__ == "__main__":

    nc_fpath = "../../stats/norm-const/all.csv"
    redo.redo_ifchange(nc_fpath)

    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple("../data/mnist/")
    rcfile = plotter.rc

    df = pd.read_csv(nc_fpath,
                     index_col=["rho"],
                     comment="#")

    with plt.rc_context(fname=rcfile):
        # fig, axs = plt.subplots(
        #     1,
        #     2,
        #     figsize=(5.5 * 0.75, 1.55),
        #     constrained_layout=True,
        #     # gridspec_kw={
        #     #     "width_ratios": [1, 0.001],
        #     # }
        # )

        fig, axs = plt.subplot_mosaic(
            [['n/rho*Z', 'Z/n'],
             ['legend', 'legend']],
            # sharey=True,
            gridspec_kw={
                'height_ratios': [1, 0.001]
            },
            figsize=(5.5, 2.5),
            constrained_layout=True,
        )

        plot_mnn(axs, df)

        fig.savefig(sys.argv[3], format="pdf")
