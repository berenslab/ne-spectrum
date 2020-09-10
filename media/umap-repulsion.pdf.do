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
    umap_init = dsrc / f"umap_knn/maxscale;f:10/umap"

    nus = [5, 500, 2000]
    n_epochs = 3000

    umap_runs = []
    umap_titles = []
    for nu in nus:
        umap_runs.append(umap_init / f"umap;n_iter:{n_epochs};nu:{nu}")
        umap_titles.append(f"\\gls{{umap}}, $\\nu={nu}$")

    tsne_runs = [
        dsrc / "affinity/stdscale;f:1e-4/tsne",
        dsrc / "affinity/stdscale;f:1e-4/tsne;late_exaggeration:2",
    ]

    titles = ["\gls{tsne}, $\\rho=2$"] + umap_titles + ["\gls{tsne}"]
    datafiles = tsne_runs[1:] + umap_runs + tsne_runs[:1]

    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotMultWithTitle(
        datafiles,
        plotname=relname,
        titles=titles,
        format="pdf",
        scalebars=0.3,
        alpha=1,
        # figheight=1.25,
        figwidth=1.33824,
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
    figs = plotter.transform()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
