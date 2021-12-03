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
from sklearn.linear_model import LinearRegression


def plot(ax, df: pd.DataFrame):

    dfg = df.groupby("algo")
    umap = dfg.get_group("umap")
    df = dfg.get_group("umapbh").copy()

    vals = []
    ns = df.index.unique()

    for n in ns:
        m1 = df.index == n
        m2 = umap.index == n

        diff = (df[m1]["extent"] - umap[m2]["extent"]).abs()
        df.loc[m1, "scale_diff"] = diff
        closest_gamma = df[m1].iloc[diff.argmin()]["gamma"]
        vals.append(1 / closest_gamma)

    line, = ax.plot(ns, vals, color="black", ls="", marker="o", clip_on=False) # mpl.colors.rgb2hex(cmap(i))

    xs1 = np.array(np.log10(ns))
    xs = xs1.reshape(-1, 1)
    ys1 = np.log10(vals)
    ys = ys1
    lin = LinearRegression().fit(xs.reshape(-1, 1), ys)
    linetxt = f"$\\log\\hat\\gamma_{{eff}} = {lin.coef_[0]:.2f} \\cdot \\log n + {lin.intercept_:.2f}$"
    # slope = -0.98759849
    # intercept = 0.63808
    preds = lin.predict(xs)
    ax.plot(ns, 10 ** preds, color="grey", alpha=0.5)
    logx = np.log10([22500])
    logy = lin.predict([logx])
    ax.text(10 ** logx[0], 10 ** logy[0], linetxt, ha="left", va="bottom")

    jnb_msc.plot.despine(ax)

    ax.set_xscale("log")
    ax.set_xlabel(r"Sample size ($n$)")
    xtix = [10000 * x for x in [1, 2, 3, 5, 7]]
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_xticks(xtix)
    ax.set_xticklabels([f"{n/1000:g}k" for n in xtix])
    ax.set_xlim(df.index[0], df.index[-1])

    ax.set_ylabel(r"Effective repulsion coefficient ($\gamma_{\mathrm{eff}}$)")
    ax.set_yscale("log")

    def fmt_y(n):
        exp = int(np.floor(np.log10(n)))
        mantissa = int(np.round(n * 10 ** -exp))
        return f"${mantissa} \\cdot 10^{{{exp}}}$" if mantissa != 1 else f"$10^{{{exp}}}$"

    ytix = [x * 10 ** -5 for x in [40, 30, 20, 10, 8, 6]]
    ax.set_yticks(ytix)
    ax.set_yticklabels([fmt_y(n) for n in ytix])


if __name__ == "__main__":

    fpath = "../stats/gamma-eff/mnist.csv"
    redo.redo_ifchange(fpath)

    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple("../data/mnist/")
    rcfile = plotter.rc

    df = pd.read_csv(fpath,
                     index_col=["n"],
                     comment="#")

    with plt.rc_context(fname=rcfile):
        fig, ax = plt.subplots(
            figsize=(5.5 * 0.5, 5.5 * 0.45),
            constrained_layout=True,
        )

        plot(ax, df)

        fig.savefig(sys.argv[3], format="pdf")
