#!/usr/bin/env python
import jnb_msc

import shutil
import sys
import inspect
import string

from umap.umap_ import find_ab_params
import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca")

    min_dist = 0.1
    spread = 1.0
    a, b = find_ab_params(spread, min_dist)

    umap = dsrc / "umap_knn/maxscale;f:10/umap;gamma:14000"
    umap_default = dsrc / f"umap_knn/spectral/maxscale;f:10/umap;a:{a};b:{b};gamma:14000"

    dpaths = [
        umap_default,
        umap,
    ]
    titles = [
        "UMAP,\nDefault,\n$\gamma=n / \\nu = 14000$",
        "UMAP,\n$\mathit{a}=b=\mathdefault{1}$\n$\gamma=n / \\nu = 14000$",
    ]

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotRow(
        dpaths,
        plotname=relname,
        titles=titles,
        format="pdf",
        scalebars=0.25,
        figwidth=2.25,
        figheight=1.45,
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

    fig.savefig(sys.argv[3], format="pdf", dpi=300)
