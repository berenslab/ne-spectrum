#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect
import string

# from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca")

    repulsion_strengths = [10 ** -i for i in range(4)]
    umap_prefix = dsrc / f"umap_knn/maxscale;f:10/"

    umap_runs = []
    titles = []
    for rs in repulsion_strengths:
        umap_runs.append(umap_prefix / f"umap;gamma:{rs}")
        titles.append(f"$\\gamma = {rs:g}$")

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotMultWithTitle(
        umap_runs, plotname=relname, titles=titles, format="pdf", scalebars=0.25,
        figwidth=6.97,
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
