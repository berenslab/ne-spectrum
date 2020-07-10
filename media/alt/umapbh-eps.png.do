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
    dsrc = Path("../../data/mnist/pca")

    exags = [2000, 3000, 5000, 7000, 10000, 20000, 30000, 50000, 70000]
    epss = [0, 1]

    umap_runs = []
    umap_titles = []
    for eps in epss:
        for exag in exags:
            umap_runs.append(
                dsrc / f"ann/rownorm/maxscale;f:10/umapbh;eps:{eps};exaggeration:{exag}"
            )
            umap_titles.append(f"$\\rho={{}}${exag}, $\\epsilon=${eps}")

    # tsne_runs = [
    #     dsrc / "affinity/stdscale;f:1e-4/tsne",
    #     dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:4",
    # ]

    # titles = ["Exag. t-SNE, $\\rho=2$"] + umap_titles + ["t-SNE"]
    titles = umap_titles
    # datafiles = tsne_runs[1:] + umap_runs + tsne_runs[:1]
    datafiles = umap_runs

    relname = sys.argv[2]
    plotter = jnb_msc.plot.ScatterMultiple(
        datafiles, plotname=relname, titles=titles, format="png", scalebars=0.3
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
