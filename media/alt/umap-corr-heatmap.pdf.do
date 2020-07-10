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


def plot_corr_heatmap(df, cmap="plasma", xtickstep=4, ytickstep=3):
    fig, ax = plt.subplots(
        figsize=(1.5, 1.5),
        constrained_layout=True,
        # gridspec_kw={"wspace": 0, "hspace": 0},
    )

    df = df.T
    df.index = df.index.set_names("rho")
    df.index = df.index.reindex([float(f) for f in df.index])[0]
    df.columns = df.columns.set_names("n")
    maxrhos_idx = df.values.argmax(axis=0)
    mat = ax.matshow(df.values, cmap=cmap)
    ax.scatter(np.arange(len(maxrhos_idx)), maxrhos_idx, marker="x", c="black")

    ax.xaxis.tick_bottom()
    ax.set_xlabel("subsamples")
    ax.set_ylabel(r"$\rho$")

    xix = np.arange(len(df.columns), step=xtickstep)
    yix = np.arange(len(df.index), step=ytickstep)
    ax.set_xticks(xix)
    ax.set_yticks(yix)
    ax.set_xticklabels([f"{n/1000:g}k" for n in df.columns[xix]])
    ax.set_yticklabels([f"{rho:g}" for rho in df.index[yix]])

    fig.colorbar(mat)
    # legend = ax.legend()
    # legend.get_frame().set_linewidth(0.4)
    return fig, ax


if __name__ == "__main__":

    dsrc_chimp = Path("../data/treutlein")

    dsrcs = [
        "../stats/umap-subsample-dcor.csv",
    ]
    redo.redo_ifchange(dsrcs)
    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple(dsrc_chimp / "phony path")
    rcfile = plotter.rc

    df = pd.read_csv(dsrcs[0], index_col="n")

    with plt.rc_context(fname=rcfile):
        fig, *_ = plot_corr_heatmap(df)

    fig.savefig(sys.argv[3], format="pdf")
    fig.savefig(sys.argv[2] + ".png")
