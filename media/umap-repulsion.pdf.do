#!/usr/bin/env python
import jnb_msc

import os
import sys
import string
import inspect

# from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../data/mnist/subsample;n:6000/pca")
    umap_prefix = dsrc / f"umap_knn/maxscale;f:10"

    nus = [5, 500, 2000]
    n_epochs = 3000

    umap_runs = [umap_prefix / f"umap;n_iter:{n_epochs}"]
    umap_titles = [r"UMAP, $\nu={}$5"]
    for nu in nus[1:]:
        # use "early exag" with umap_prefix / "umap" / f"umap;..."
        # and add ", EE" to the title
        umap_runs.append(umap_prefix / "umap" /
                         f"umap;n_iter:{n_epochs};nu:{nu}")
        umap_titles.append(f"UMAP,\n$\\nu={{}}${nu}, EE")

    umap_runs2 = []
    umap_titles2 = []
    for nu in nus[1:]:
        gamma = 5 / nu
        umap_runs2.append(umap_prefix / f"umap;n_iter:{n_epochs};nu:{nu};gamma:{gamma}")
        umap_titles2.append(f"UMAP,\n$\\nu={{}}${nu}, $\\gamma={{}}${gamma}")

    tsne_runs = [
        dsrc / "affinity/stdscale;f:1e-4/tsne",
        dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:2",
    ]

    # phony = tsne_runs[0]      # doesn't matter what this is
    # titles = ["t-SNE, $\\rho=$2"] + umap_titles + ["t-SNE"] \
    #     + ["remove", "remove", umap_titles2[0], umap_titles2[1], "remove"]
    # datafiles = tsne_runs[1:] + umap_runs + tsne_runs[:1] \
    #     + [phony, phony, umap_runs2[0], umap_runs2[1], phony]
    letters = "abcde" + "xxfgx"

    relname = sys.argv[2]

    datafiles = tsne_runs[1:] + umap_runs + umap_runs2 + tsne_runs[:1]
    titles = ["t-SNE, $\\rho=$2"] + umap_titles + umap_titles2 + ["t-SNE"]

    plotter = jnb_msc.plot.ScatterMultiple(
        datafiles,
        plotname=relname,
        titles=titles,
        lettering=letters,
        layout=(2,5),
        format="pdf",
        scalebars=0.3,
        alpha=1,
        figheight=1.25,
    )
    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in plotter.__class__.mro()]
            if hasattr(mod, "__file__")  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
        ]
    )

    datadeps = plotter.get_datadeps()

    jnb_msc.redo.redo_ifchange(list(filedeps) + datadeps)
    plotter.load()
    # fig, axs = plotter.transform()
    # axs[1,0].remove()
    # axs[1,1].remove()
    # axs[1,4].remove()
    fig = plt.figure(figsize=(5.5, 2.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 5) # , wspace=0.1, hspace=.25)
    axs = []
    axs.append(fig.add_subplot(gs[ : , 0]))
    axs.append(fig.add_subplot(gs[ : , 1]))
    axs.append(fig.add_subplot(gs[ :1, 2]))
    axs.append(fig.add_subplot(gs[ :1, 3]))
    axs.append(fig.add_subplot(gs[1: , 2]))
    axs.append(fig.add_subplot(gs[1: , 3]))
    axs.append(fig.add_subplot(gs[ : , 4]))

    with plt.rc_context(fname=plotter.rc):
        for ax, dat, title, letter in zip(axs, plotter.data, titles, string.ascii_lowercase):
            ax.scatter(
                dat[:, 0],
                dat[:, 1],
                c=plotter.labels,
                # alpha=self.alpha,
                rasterized=True,
            )
            ax.set_title(title)
            plotter.add_lettering(ax, letter)
            plotter.add_scalebar(ax, 0.3)
            jnb_msc.plot.set_aspect_center(ax)


        # plotter.save()
        fig.savefig(sys.argv[3], format="pdf")

    # link to the result
    # os.link(plotter.outdir / relname, sys.argv[3])
