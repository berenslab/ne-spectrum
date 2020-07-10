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


def plot_z(df):
    fig, ax = plt.subplots(figsize=(5.5 / 2, 1))

    nz = df["Z/N"] ** -1
    # ax.plot(df["rho"], nz, label=r"$\frac{N}{Z}$")
    ax.plot(df["rho"], nz / df["rho"])
    ax.set_xlabel(r"Exaggeration factor ($\rho$)")
    ax.set_ylabel(r"${n}/(Z\rho)$")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.legend()
    return fig, ax


if __name__ == "__main__":

    dsrc_chimp = Path("../../data/treutlein")

    zstats = "../../stats/z_stats.csv"
    redo.redo_ifchange(zstats)
    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple(dsrc_chimp / "phony path")
    rcfile = plotter.rc

    df = pd.read_csv(zstats)

    with plt.rc_context(fname=rcfile):
        fig, *_ = plot_z(df)
    fig.savefig(sys.argv[3], format="png", bbox_inches="tight")
    # fig.savefig(sys.argv[2] + ".pdf", format="pdf", bbox_inches="tight")
