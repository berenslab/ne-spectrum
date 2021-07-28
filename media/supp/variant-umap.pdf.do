#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect
import string

from umap.umap_ import find_ab_params
import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca")
    knn_prefix = "ann/spnorm"

    min_dist = 0.1
    spread = 1.0
    a, b = find_ab_params(spread, min_dist)

    umap = dsrc / "umap_knn/maxscale;f:10/umap"
    umap_default = dsrc / f"umap_knn/spectral/maxscale;f:10/umap;a:{a};b:{b}"
    umap_knn = dsrc / knn_prefix / "maxscale;f:10/umap"
    umap_knn_eps = dsrc / knn_prefix / "maxscale;f:10/umap;eps:1"

    dpaths = [
        umap_default,
        umap,
        umap_knn,
        umap_knn_eps,
    ]
    titles = [
        "Default UMAP\n",
        "UMAP,\n$\mathit{a}=b=\mathdefault{1}$",
        "UMAP,\n$a=b=\mathdefault{1}$, kNN affin.",
        "UMAP,\n$a=b=\mathdefault{1}$, kNN affin., $\epsilon=\mathdefault{1}$",
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
        figheight=1.5,
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
    # flip the default UMAP with LE init
    # plotter.data[0][:, 1] *= -1
    fig, axs = plotter.transform()

    # with plt.rc_context(fname=plotter.rc):
    #     gs = axs[0, 0].get_gridspec()
    #     ax = fig.add_subplot(gs[1, :])
    #     ax.set_title("UMAP\n")
    #     ax.set_axis_off()

    fig.savefig(sys.argv[3], format="pdf")
