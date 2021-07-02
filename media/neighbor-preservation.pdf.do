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


def plot_mnn(ax, df):
    ax.plot(df.index, df["mnist"],           label="MNIST")
    ax.plot(df.index, df["treutlein"],       label="Chimp organoid")
    ax.plot(df.index, df["treutlein_409b2"], label="Human organoid")
    ax.plot(df.index, df["famnist"],         label="Fashion MNIST")
    ax.plot(df.index, df["kuzmnist"],        label="Kuzushiji MNIST")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$")

    ax.set_ylabel("MNN")
    ax.set_ylim(bottom=0)

    ax.legend()

    return ax


if __name__ == "__main__":

    mnn_fpath = "../stats/neighborhood_ratios.csv"
    # redo.redo_ifchange(mnn_fpath)
    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple("../data/mnist/")
    rcfile = plotter.rc

    df = pd.read_csv(mnn_fpath,
                     index_col="rho",
                     comment="#")

    with plt.rc_context(fname=rcfile):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(2.75, 1.55),
            constrained_layout=True,
        )

        plot_mnn(ax, df)

        fig.savefig(sys.argv[3], format="pdf")
