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
    dsrc = Path("../../data/mnist")
    umap_init = dsrc / f"umap_knn/maxscale;f:10/umap"

    nus = [5, 500, 2000]
    n_epochs = 3000

    umap_runs = []
    umap_titles = []
    for n in range(5000, 70000, 5000):
        for rho in [1, 4]:
            tsne_ = "tsne;save_iter_freq:999" + (
                "" if rho == 1 else f";late_exaggeration:{rho}"
            )
            umap_runs.append(
                dsrc / f"subsample;n:{n}/pca/affinity/stdscale;f:1e-4" / tsne_
            )
            umap_titles.append(f"t-SNE, $\\rho={{}}${rho}")

    titles = ["Exag. t-SNE, $\\rho=2$"] + umap_titles + ["t-SNE"]
    # datafiles = tsne_runs[1:] + umap_runs + tsne_runs[:1]
    datafiles = umap_runs

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
