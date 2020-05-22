#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect
import dcor
import scipy

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from pathlib import Path
from itertools import cycle
from matplotlib.legend_handler import HandlerTuple


def plot_correlation(rhos, corrs, dsrc_names=None, alpha=1):
    fig, ax = plt.subplots(figsize=(5.5 / 2, 1))
    linestyles = cycle(["solid", "dashed", "dashdot"])
    c_fa2 = "tab:blue"
    c_umap = "tab:orange"
    ax.set_zorder(6)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    legend_handles = []
    legend_labels = []
    for idx, corr in enumerate(corrs):
        c1, c2 = corr.T
        dname = "" if dsrc_names is None else (str(dsrc_names[idx]) + " ")

        ls = next(linestyles)
        line_fa2 = ax.plot(
            rhos, c1, label=dname + "FA2", alpha=alpha, zorder=5, ls=ls, c=c_fa2,
        )
        m_ix = np.argmax(c1)
        ax.scatter(
            rhos[m_ix], c1[m_ix], alpha=alpha, marker="o", s=10, c=c_fa2, zorder=6
        )
        line_umap = ax.plot(
            rhos, c2, label=dname + "UMAP", alpha=alpha, zorder=5, ls=ls, c=c_umap,
        )
        m_ix = np.argmax(c2)
        ax.scatter(
            rhos[m_ix], c2[m_ix], alpha=alpha, marker="o", s=10, c=c_umap, zorder=6,
        )

        line_fa2 = line_fa2[0]
        line_umap = line_umap[0]

        handle = mpl.lines.Line2D([], [], ls=ls, color="xkcd:dark grey")
        # legend_handles.append((line_fa2, line_umap))
        legend_handles.append(handle)
        legend_labels.append(dname.strip())

        # ax.plot(rhos[25:35], c1[25:35], marker="o", c=line_fa2.get_color())
        # ax.plot(rhos[2:6], c1[2:6], marker="o", c=line_umap.get_color())

    ax.set_ylim(0.8, 1)
    ax.set_xlim(1, 100)
    ax.set_ylabel(r"Distance correlation")
    ax.set_xlabel(r"Exaggeration factor ($\rho$)")
    ax.set_xscale("log")

    legend = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        fancybox=False,
        fontsize="small",
        loc="best",
        framealpha=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    legend.get_frame().set_linewidth(0.4)
    # draw lines for the rho values that are the most common choices
    # for UMAP and FA2
    _, _, ymin, ymax = ax.axis()
    ax.set_ylim(ymin, ymax)
    ax.plot(
        [4, 4],
        [ymin, ymax],
        color="lightgrey",
        lw=plt.rcParams["axes.linewidth"],
        alpha=0.5,
    )
    ax.plot(
        [30, 30],
        [ymin, ymax],
        color="lightgrey",
        lw=plt.rcParams["axes.linewidth"],
        alpha=0.5,
    )

    ax.text(30, 1, "FA2", color=c_fa2)
    ax.text(4, 1, "UMAP", color=c_umap)

    return fig, ax


if __name__ == "__main__":

    dsrc_chimp = Path("../data/treutlein")

    dsrcs = [
        "../stats/mnist_corr.csv",
        "../stats/thuman_corr.csv",
        "../stats/tchimp_corr.csv",
    ]
    redo.redo_ifchange(dsrcs)
    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple(dsrc_chimp / "phony path")
    rcfile = plotter.rc

    dfs = [pd.read_csv(ds) for ds in dsrcs]
    rhos = dfs[0]["rho"]
    corrs = [df[["fa2", "umap"]].values for df in dfs]

    with plt.rc_context(fname=rcfile):
        fig, *_ = plot_correlation(
            rhos, corrs, dsrc_names=["MNIST", "Human organoid", "Chimp organoid"]
        )
    fig.savefig(sys.argv[3], format="pdf", bbox_inches="tight")
    fig.savefig(sys.argv[2] + ".png", format="png", bbox_inches="tight")
