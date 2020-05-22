#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patches import BoxStyle
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

from pathlib import Path


def draw_arrow(
    fig,
    axesA,
    axesB,
    xyA=(0.5, 1),
    xyB=(0.5, 0.5),
    shrinkA=15,
    shrinkB=33,
    head_width=0.15,
    head_length=0.15,
):
    arrowstyle = f"->,head_width={head_width},head_length={head_length}"
    arrow = patches.ConnectionPatch(
        xyA=xyA,
        xyB=xyB,
        coordsA="axes fraction",
        coordsB="axes fraction",
        axesA=axesA,
        axesB=axesB,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        color="black",
        linewidth=0.4,
        arrowstyle=arrowstyle,
    )
    fig.add_artist(arrow)


def plot_spectrum(
    spectral, fa2, umap, tsne, tsne4, tsne30, tsne100, tsnehalf, labels, alpha=0.3
):
    width_inch = 5.5  # text (and thus fig) width of nips
    rows = 2
    cols = 6
    box_inch = width_inch / cols
    fig, axs = plt.subplots(
        rows,
        cols,
        # ugly hack to have the figure have as many pixels as the
        # other ones (3,300)
        figsize=(width_inch, rows * box_inch),
        constrained_layout=False,
    )
    fig.subplots_adjust(0, 0, 1, 1)
    gs = axs[0, 0].get_gridspec()

    axs[0, 0].set_title("Laplacian Eigenmaps")
    axs[0, 0].scatter(
        spectral[:, 0], spectral[:, 1], c=labels, alpha=alpha, rasterized=True
    )
    axs[0, 0].set_xlabel("Eig 1", labelpad=2)
    sp_ylbl = axs[0, 0].set_ylabel("Eig 2", labelpad=2, va="baseline")
    axs[0, 0].set_zorder(2)
    axs[0, 3].set_title("t-SNE ($\\rho = {}$1)")
    axs[0, 3].scatter(tsne[:, 0], tsne[:, 1], c=labels, alpha=alpha, rasterized=True)
    axs[0, 3].set_zorder(3)
    axs[1, 1].set_title("ForceAtlas2")
    axs[1, 1].scatter(fa2[:, 0], fa2[:, 1], c=labels, alpha=alpha, rasterized=True)
    axs[1, 2].set_title("UMAP")
    axs[1, 2].scatter(umap[:, 0], umap[:, 1], c=labels, alpha=alpha, rasterized=True)

    # remove unused axes
    axs[1, 0].remove()
    axs[1, 3].remove()
    axs[1, 4].remove()
    axs[1, 5].remove()

    axs[0, 1].remove()
    axs[0, 2].remove()
    axs[0, 4].remove()
    axs[0, 5].remove()
    # fig.set_size_inches(width_inch, rows * box_inch)

    # add t-sne with exaggeration
    gs_exag = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1:3])
    exag_axs = []
    for title, data, g in zip(
        [r"$ρ = {}$100", r"$\rho = {}$30", r"$\rho = {}$4"],
        [tsne100, tsne30, tsne4],
        gs_exag,
    ):
        ax = fig.add_subplot(g, zorder=5)
        exag_axs.append(ax)
        ax.set_title(title)
        ax.scatter(data[:, 0], data[:, 1], c=labels, alpha=alpha, rasterized=True)

    # create this gridspec for three plots to enfore the same size of
    # the previous plots
    gs_half = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 4:])
    ax = fig.add_subplot(gs_half[0], zorder=5)
    ax.set_title(r"$\rho = \mathdefault{\frac{1}{2}}$")
    ax.scatter(tsnehalf[:, 0], tsnehalf[:, 1], c=labels, alpha=alpha, rasterized=True)

    arrstyle = mpl.patches.ArrowStyle("<->", head_length=0.4, head_width=0.4)
    _, tops, lefts, rights = gs_exag.get_grid_positions(fig)
    pos = gs_exag[0].get_position(fig)
    upper_left = lefts[0], tops[0]
    upper_right = rights[-1], tops[0]
    arr = mpl.patches.FancyArrowPatch(
        upper_left,
        upper_right,
        lw=0.4,
        shrinkA=0,
        shrinkB=0,
        arrowstyle="<->,head_width=1.5,head_length=1.5",
    )
    fig.add_artist(arr)

    eps = abs(lefts[0] - rights[-1]) * 0.03
    fig.text(
        upper_left[0] + eps,
        upper_left[1] * 1.01,
        "Stronger attraction",
        va="bottom",
        ha="left",
    )
    fig.text(
        upper_right[0] - eps,
        upper_right[1] * 1.01,
        "Stronger repulsion",
        va="bottom",
        ha="right",
    )

    # add the lower eigenvecs
    gs_spectral = gs[:, 0].subgridspec(4, 2)
    cmap = plt.get_cmap("tab20", lut=np.unique(labels).shape[0])
    title_fmt = "Eigs {}/{}"
    label_fmt = "Eig {}"

    # only show those three digits
    mask = np.isin(labels, [4, 7, 9])
    # transform all digits to color space and then subselect in order
    # to keep the same colors
    clrs = cmap(labels)[mask]
    ax = fig.add_subplot(gs_spectral[2, 0], zorder=5, label="spectral_sub")
    i, j = 7, 9
    # ax.set_xlabel(
    #     title_fmt.format(i, j),
    #     fontsize=plt.rcParams["axes.titlesize"],
    #     labelpad=plt.rcParams["axes.titlepad"],
    # )
    ax.set_xlabel(label_fmt.format(i), labelpad=2)
    ax.set_ylabel(label_fmt.format(j), labelpad=2, va="baseline")
    ax.scatter(
        spectral[mask, i - 1],
        spectral[mask, j - 1],
        c=clrs,
        alpha=alpha,
        rasterized=True,
    )

    mask = np.isin(labels, [3, 5, 8])
    clrs = cmap(labels)[mask]
    ax = fig.add_subplot(gs_spectral[2, 1], zorder=5, label="spectral_sub")
    i, j = 12, 9
    ax.set_xlabel(label_fmt.format(i), labelpad=2)
    # ax.set_ylabel(label_fmt.format(j))
    # ax.set_xlabel(
    #     title_fmt.format(i, j),
    #     fontsize=plt.rcParams["axes.titlesize"],
    #     labelpad=plt.rcParams["axes.titlepad"],
    # )
    ax.scatter(
        spectral[mask, i - 1],
        -spectral[mask, j - 1],
        c=clrs,
        alpha=alpha,
        rasterized=True,
    )

    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
        if ax.get_label() != "arrow_bg":
            jnb_msc.plot.set_aspect_center(ax)
        if ax.get_label() == "spectral_sub":
            pos = ax.get_position()
            y_len = pos.y1 - pos.y0
            # hacky way to determine the whitespace in-between the
            # spectral subvectors
            hspace = y_len * gs_spectral.get_subplot_params().wspace
            # place the axis below the full spectral embedding
            pos.y1 = sp_ylbl.get_position()[1]
            pos.y0 = pos.y1 - y_len
            ax.set_position(pos)

    draw_arrow(fig, axesA=axs[1, 1], axesB=exag_axs[1])
    draw_arrow(fig, axesA=axs[1, 2], axesB=exag_axs[2], shrinkA=12, shrinkB=25)

    # add the arrow that indicates the repulsion spectrum
    pos = axs[0, 0].get_position()
    # x coords start and end
    startpoint = pos.x1
    endpoint = gs_half[1].get_position(fig).x0 - startpoint
    y_frac = (pos.y0 + pos.y1) / 2
    bg_ax = fig.add_axes([startpoint, 0, endpoint, 1], zorder=1, label="arrow_bg")
    bg_ax.set_axis_off()
    # bg_ax.arrow(0, .75, 1, 0)
    draw_arrow(
        fig,
        bg_ax,
        bg_ax,
        xyA=(0, y_frac),
        xyB=(1, y_frac),
        shrinkA=0,
        shrinkB=0,
        head_width=0.2,
        head_length=0.2,
    )

    ## obviously this is super specific to the plot at hand
    for i in range(10):
        mask = labels == i
        box = TextArea(str(i))

        mean = tsne[mask].mean(axis=0)
        abox = AnnotationBbox(box, mean, frameon=False, pad=0)
        axs[0, 3].add_artist(abox)  # tsne axis

    return fig, axs


if __name__ == "__main__":
    dsrc = Path("../data/mnist/pca")

    spectral = dsrc / "ann/spectral/data.npy"
    fa2 = dsrc / "ann/fa2/data.npy"
    umap = dsrc / "umap_knn/maxscale;f:10/umap/data.npy"

    tsne = dsrc / "affinity/stdscale;f:1e-4/tsne/data.npy"
    tsne4 = dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:4/data.npy"
    tsne30 = (
        dsrc
        / "affinity/stdscale;f:1e-4/tsne;early_exaggeration:30;late_exaggeration:30/data.npy"
    )
    tsne100 = (
        dsrc
        / "affinity/stdscale;f:1e-4/tsne;early_exaggeration:100;late_exaggeration:100;learning_rate:1/data.npy"
    )
    tsnehalf = dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:0.5/data.npy"

    tsnes = [tsne, tsne4, tsne30, tsne100, tsnehalf]

    datafiles = [spectral, fa2, umap] + tsnes
    # abuse the plotter object to get the rc and labels
    plotter = jnb_msc.plot.ScatterMultiple(datafiles)
    redo.redo_ifchange(datafiles + [plotter.labelname, plotter.rc])
    # plotter.load()
    rcfile = plotter.rc
    labels = np.load(plotter.labelname)

    data = [np.load(f) for f in datafiles]

    # flip the spectral embedding to align with the other embeddings
    # (on MNIST)
    data[0] *= -1

    with plt.rc_context({"font.size": 4.25, "axes.titlesize": "larger"}, fname=rcfile):
        fig, *_ = plot_spectrum(*data, labels=labels)
    fig.savefig(sys.argv[3], format="pdf", bbox_inches="tight")
    # link to the result
    # os.link(plotter.outdir / relname, sys.argv[3])
