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
    dsrc = Path("../../data/mnist/subsample;n:6000/pca")
    umap_init = dsrc / f"umap_knn/maxscale;f:10"

    nus = [5, 500, 2000]
    n_epochs = 3000

    umap_runs = []
    umap_titles = []
    umap_runs.append(umap_init / f"umap")
    umap_titles.append(f"UMAP")

    tsne_runs = [
        dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:2",
        dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:4",
        dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:6",
    ]

    titles = ["$\\rho=2$", "$\\rho=4$", "$\\rho=6$"] + umap_titles
    datafiles = tsne_runs + umap_runs

    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotRow(
        datafiles,
        plotname=relname,
        titles=titles,
        format="pdf",
        scalebars=0.3,
        alpha=1,
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
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
