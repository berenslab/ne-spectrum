#!/usr/bin/env python
import jnb_msc

import os
import sys
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

    umap_runs = []
    umap_titles = []
    for nu in nus:
        umap_runs.append(umap_prefix / f"umap;n_iter:{n_epochs};nu:{nu}")
        umap_titles.append(f"UMAP, $\\nu={{}}${nu}")

    umap_runs2 = []
    umap_titles2 = []
    for nu in nus[1:]:
        gamma = 5 / nu
        umap_runs2.append(umap_prefix / f"umap;n_iter:{n_epochs};nu:{nu};gamma:{gamma}")
        umap_titles2.append(f"UMAP, $\\nu={{}}${nu}\n$\\gamma={{}}${gamma}")

    tsne_runs = [
        dsrc / "affinity/stdscale;f:1e-4/tsne",
        dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:2",
    ]

    phony = tsne_runs[0]      # doesn't matter what this is
    titles = ["t-SNE, $\\rho=$2"] + umap_titles + ["t-SNE"] \
        + ["remove", "remove", umap_titles2[0], umap_titles2[1], "remove"]
    datafiles = tsne_runs[1:] + umap_runs + tsne_runs[:1] \
        + [phony, phony, umap_runs2[0], umap_runs2[1], phony]
    letters = "abcde" + "xxfgx"

    relname = sys.argv[2]
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
    fig, axs = plotter.transform()
    axs[1,0].remove()
    axs[1,1].remove()
    axs[1,4].remove()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
