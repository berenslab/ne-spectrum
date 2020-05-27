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


def plot_correlation(rhos, corrs, fig=None, ax=None, dsrc_names=None, alpha=1):
    if fig is None:
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
            rhos[m_ix],
            c1[m_ix],
            alpha=alpha,
            marker="o",
            s=10,
            c=c_fa2,
            clip_on=False,
            zorder=6,
        )
        line_umap = ax.plot(
            rhos, c2, label=dname + "UMAP", alpha=alpha, zorder=5, ls=ls, c=c_umap,
        )
        m_ix = np.argmax(c2)
        ax.scatter(
            rhos[m_ix],
            c2[m_ix],
            alpha=alpha,
            marker="o",
            s=10,
            c=c_umap,
            clip_on=False,
            zorder=6,
        )

        line_fa2 = line_fa2[0]
        line_umap = line_umap[0]

        handle = mpl.lines.Line2D([], [], ls=ls, color="xkcd:dark grey")
        # legend_handles.append((line_fa2, line_umap))
        legend_handles.append(handle)
        legend_labels.append(dname.strip())

        # ax.plot(rhos[25:35], c1[25:35], marker="o", c=line_fa2.get_color())
        # ax.plot(rhos[2:6], c1[2:6], marker="o", c=line_umap.get_color())

    ax.set_ylim(0.9, 1)
    ax.set_xlim(1, 100)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100])
    t = list(range(2, 10, 1)) + list(range(20, 100, 10))
    ax.set_xticks(t, minor=True)

    ax.set_ylabel(r"Distance correlation")
    ax.set_xlabel(r"Exaggeration factor ($\rho$)")

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
    # _, _, ymin, ymax = ax.axis()
    # ax.set_ylim(ymin, ymax)
    # ax.plot(
    #     [4, 4],
    #     [ymin, ymax],
    #     color="lightgrey",
    #     lw=plt.rcParams["axes.linewidth"],
    #     alpha=0.5,
    # )
    # ax.plot(
    #     [30, 30],
    #     [ymin, ymax],
    #     color="lightgrey",
    #     lw=plt.rcParams["axes.linewidth"],
    #     alpha=0.5,
    # )

    ax.text(30, 1, "FA2", color=c_fa2)
    ax.text(4, 1, "UMAP", color=c_umap)

    return fig, ax


def plot_correlation_subsample(df, fig=None, ax=None, color="xkcd:dark grey"):
    if fig is None:
        fig, ax = plt.subplots(
            figsize=(1, 1),
            constrained_layout=True,
            # gridspec_kw={"wspace": 0, "hspace": 0},
        )

    rhos = np.sort(df["rho"].unique())
    # rhos = [4]
    for rho in rhos:
        m = df["rho"] == rho
        dfm = df[m]
        nrhoz = dfm["Z/N"] ** -1 / rho
        ax.plot(dfm["n"], nrhoz, label=f"$\\rho = {{}}${rho}", color=color)
        x = dfm["n"].tail(1).item()
        y = nrhoz.tail(1).item()
        ax.annotate(
            f" $\\rho = {{}}${rho}",
            xy=(x, y),
            va="center",
            ha="left",
            annotation_clip=False,
        )

    ax.set_xlabel("Sample size ($n$)", va="top")
    ax.set_ylabel(r"${n}/(\rho Z)$")
    ax.set_xlim(5000, 70000)

    ax.set_yscale("log")
    ax.set_xscale("log")
    # ax.set_xticks([5 * 10 ** 3, 10 ** 4, 7 * 10 ** 4])
    xtix = [5000, 70000]
    ax.set_xticks(xtix)
    ax.set_xticklabels([f"{n/1000:g}k" for n in xtix])
    # xix = np.arange(len(dfm["n"]), step=4)
    # ax.set_xticks(dfm["n"].values[xix])
    # ax.set_xticklabels([f"{n/1000:g}k" for n in dfm["n"].values[xix]])
    ax.set_yticks([10 ** -i for i in range(1, 5)])

    ax.set_aspect("equal")
    jnb_msc.plot.despine(ax)
    # legend = ax.legend(loc="best", fontsize="xx-small")
    # legend.get_frame().set_linewidth(0.4)
    return fig, ax


def plot_z(df, fig=None, ax=None, color="xkcd:dark grey"):
    if fig is None:
        fig, ax = plt.subplots(figsize=(5.5 / 2, 1))

    nz = df["Z/N"] ** -1
    # ax.plot(df["rho"], nz, label=r"$\frac{N}{Z}$")
    ax.plot(df["rho"], nz / df["rho"], color=color)
    ax.set_xlabel(r"Exag. factor ($\rho$)")
    ax.set_ylabel(r"${n}/(\rho Z)$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10 ** 0, 10 ** 2)
    ax.set_ylim(10 ** -7, 10 ** -2)
    ax.set_xticks([10 ** i for i in range(3)])
    ax.set_yticks([10 ** -i for i in range(2, 8)])
    ax.set_aspect("equal")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return fig, ax


def plot_corr_heatmap(
    df, fig=None, ax=None, cax=None, cmap="plasma", xtickstep=6, ytickstep=6
):
    if fig is None:
        fig, [cax, ax] = plt.subplots(
            2,
            gridspec_kw={"height_ratios": [0.1, 0.9]},
            figsize=(1.5, 1.5),
            constrained_layout=True,
        )

    df = df.T[::-1]
    df.index = df.index.set_names("rho")
    df.index = df.index.reindex([float(f) for f in df.index])[0]
    df.columns = df.columns.set_names("n")
    maxrhos_idx = df.values.argmax(axis=0)

    mat = ax.matshow(df.values, cmap=cmap, vmin=0.95, vmax=1)
    ax.plot(np.arange(len(maxrhos_idx)), maxrhos_idx, c="xkcd:dark grey")
    ax.set_aspect("auto")

    ax.xaxis.tick_bottom()
    ax.set_xlabel("Sample size ($n$)")
    ax.set_ylabel(r"Exag. factor ($\rho$)")

    # xix = np.arange(len(df.columns), step=xtickstep)
    xix = [0, 6, 13]
    yix = np.arange(len(df.index), step=ytickstep)
    ax.set_xticks(xix)
    ax.set_yticks(yix)
    ax.set_xticklabels([f"{n/1000:g}k" for n in df.columns[xix]])
    ax.set_yticklabels([f"{rho:g}" for rho in df.index[yix]])

    cbar = fig.colorbar(
        mat, cax=cax, ax=ax, orientation="vertical", aspect=100, shrink=0.75
    )
    cbar.set_label("Dist. corr.", labelpad=-4)
    return fig, ax


if __name__ == "__main__":

    dsrc_chimp = Path("../data/treutlein")

    dsrcs = [
        "../stats/mnist_corr.csv",
        "../stats/thuman_corr.csv",
        "../stats/tchimp_corr.csv",
    ]
    zstats = "../stats/z_stats.csv"
    heatmap_df = "../stats/umap-subsample-dcor.csv"
    subsample_df = "../stats/z_subsamples.csv"
    redo.redo_ifchange(dsrcs + [heatmap_df, zstats, subsample_df])
    # abuse the plotter object to get the rc
    plotter = jnb_msc.plot.ScatterMultiple(dsrc_chimp / "phony path")
    rcfile = plotter.rc

    dfs = [pd.read_csv(ds) for ds in dsrcs]
    rhos = dfs[0]["rho"]
    corrs = [df[["fa2", "umap"]].values for df in dfs]

    df_heatmap = pd.read_csv(heatmap_df, index_col="n")
    df_zstats = pd.read_csv(zstats)
    df_subsample = pd.read_csv(subsample_df)

    with plt.rc_context(fname=rcfile):
        fig, axs = plt.subplots(
            1,
            4,
            figsize=(5.5, 1.25),
            gridspec_kw={"width_ratios": [2, 0.66, 1, 1], "wspace": 0.1},
            constrained_layout=True,
        )
        gs = axs[2].get_gridspec()
        # axs[2].remove()
        # hm_gs = gs[2].subgridspec(1, 2, width_ratios=[1, 0.075], wspace=0.05)
        # hm_ax = fig.add_subplot(hm_gs[0]), fig.add_subplot(hm_gs[1])
        hm_ax = axs[2]
        plot_correlation(
            rhos,
            corrs,
            fig=fig,
            ax=axs[0],
            dsrc_names=["MNIST", "Human organoid", "Chimp organoid"],
        )
        plot_z(df_zstats, fig=fig, ax=axs[1])
        plot_corr_heatmap(df_heatmap, fig=fig, ax=hm_ax)
        plot_correlation_subsample(df_subsample, fig=fig, ax=axs[3])

        fd = jnb_msc.plot.ScatterMultiple.get_letterdict()
        fd["ha"] = "right"
        pad = plt.rcParams["axes.titlepad"]
        for ax, ltr in zip([axs[0], axs[1], hm_ax, axs[3]], "abcd"):
            ax.set_title(ltr + "    ", fontdict=fd, loc="left", pad=pad)
            # trans = ax.yaxis.get_label().get_transform()
            # ax.text(0, 1, ltr, fontdict=fd, transform=trans)

    # really bad and non-portable way to get a pdf that has minus
    # signs!!!
    alt_out = Path(sys.argv[2])
    alt_out = alt_out.with_name(alt_out.stem + ".svg")
    fig.savefig(alt_out, format="svg", bbox_inches="tight")
    subprocess.call(["rsvg-convert", "-f", "pdf", "-o", sys.argv[3], alt_out])
    # fig.savefig(sys.argv[3], format="pdf", bbox_inches="tight")
    fig.savefig(sys.argv[2] + ".png", format="png", bbox_inches="tight")
