"""Some plotting funcitons that only have a single purpose."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import cycle
from matplotlib.legend_handler import HandlerTuple


def plot_correlation_rhos(rhos, corrs, ax, alpha=1, lo_exag=4, hi_exag=30):
    """Plots graphs showing the distance correlation between FA2/UMAP
    and t-SNE with various exaggeration values."""

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

        ls = next(linestyles)
        line_fa2 = ax.plot(
            rhos, c1, label="FA2", alpha=alpha, zorder=5, ls=ls, c=c_fa2,
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
            rhos, c2, label="UMAP", alpha=alpha, zorder=5, ls=ls, c=c_umap,
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
    )
    legend.get_frame().set_linewidth(0.4)
    # draw lines for the rho values that are the most common choices
    # for UMAP and FA2
    _, _, ymin, ymax = ax.axis()
    ax.set_ylim(ymin, ymax)
    if lo_exag is not None:
        ax.plot(
            [lo_exag, lo_exag],
            [ymin, ymax],
            color="lightgrey",
            lw=plt.rcParams["axes.linewidth"],
            alpha=0.5,
        )
        # ax.text(lo_exag, 1, "UMAP", color=c_umap)
    if hi_exag is not None:
        ax.plot(
            [hi_exag, hi_exag],
            [ymin, ymax],
            color="lightgrey",
            lw=plt.rcParams["axes.linewidth"],
            alpha=0.5,
        )
        # ax.text(hi_exag, 1, "FA2", color=c_fa2)

    return ax
