#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca")
    knn_prefix = "ann/spnorm"
    # use "aann/spnorm/spsym" to have an easier textual description

    tsne_default = dsrc / "affinity/stdscale;f:1e-4/tsne"
    tsne_knn = dsrc / knn_prefix / "stdscale;f:1e-4/tsne"

    dpaths = [
        tsne_default,
        tsne_knn,
    ]
    titles = [
        "Default t-SNE\n",
        "t-SNE,\nkNN affin. ($k={}$15)",
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
        figwidth=2.25,
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
    #

    # with plt.rc_context(fname=plotter.rc):
    #     gs = axs[0, 0].get_gridspec()
    #     ax = fig.add_subplot(gs[1, :])
    #     ax.set_title("UMAP\n")
    #     ax.set_axis_off()
    fig.savefig(sys.argv[3], format="pdf")
