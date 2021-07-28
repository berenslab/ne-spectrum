#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca")

    fa2 = dsrc / "ann/fa2;use_degrees:0"
    fa2_degrep = dsrc / "ann/fa2"

    dpaths = [
        fa2_degrep,
        fa2,
    ]
    titles = [
        "FA2,\nRepulsion by degree",
        "FA2,\nFixed repulsion",
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

    # with plt.rc_context(fname=plotter.rc):
    #     gs = axs[0, 0].get_gridspec()
    #     ax = fig.add_subplot(gs[1, :])
    #     ax.set_title("UMAP\n")
    #     ax.set_axis_off()

    fig.savefig(sys.argv[3], format="pdf")
