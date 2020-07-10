#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/gauss_devel")
    init = "random"

    umap = dsrc / "umap_knn" / init / "maxscale;f:10" / "umap"
    tsnes = [
        dsrc / "affinity" / init / f"stdscale;f:1e-4/tsnestage;exaggeration:{rho}/"
        for rho in [4, 3, 2, 1.5]
    ]
    datafiles = [umap] + tsnes

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotRow(
        datafiles, plotname=relname, titles=None, format="png", alpha=0.5
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
    plotter.transform()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
