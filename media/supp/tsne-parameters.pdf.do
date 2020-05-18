#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca/")

    datafiles = [
        dsrc / "affinity/stdscale;f:1e-4/tsne",
        dsrc / "affinity/stdscale;f:1e-4/tsne;early_exaggeration:1/",
        dsrc / "affinity/stdscale;f:25/tsne",
        # dsrc / "stdscale;f:1e-4/affinity/tsne;learning_rate:200;early_exaggeration:1/",
        # dsrc / "stdscale;f:25/affinity/tsne;learning_rate:200",
    ]

    titles = [
        "With early exaggeration\nInitial std${}={}$0.0001",
        "Without early exaggeration\nInitial std${}={}$0.0001",
        "With early exaggeration\nInitial std${}={}$25",
    ]

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotRow(
        datafiles, plotname=relname, titles=titles, format="png", scalebars=0.2
    )
    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in plotter.__class__.mro()]
            if hasattr(mod, "__file__")  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
        ]
    )

    datadeps = plotter.get_datadeps()

    redo.redo_ifchange(list(filedeps) + datadeps)
    plotter.load()
    fig, axs = plotter.transform()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
