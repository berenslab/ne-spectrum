#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path
from itertools import product

if __name__ == "__main__":
    dsrc = Path("../../data/gauss_devel/ann/")
    datapaths = []
    titles = []
    init = ["pca", "random"]
    stds = [1, 2, 3, 4]

    for std, ini in product(stds, init):
        datapaths += [
            dsrc / ini / f"stdscale;f:{10**std}" / "fa2",
            dsrc / ini / f"stdscale;f:{10**std}" / "fa2;use_degrees:0",
        ]
        titles += [
            f"deg. rep., std {10**std}, init {ini}",
            f"fix. rep., std {10**std}, init {ini}",
        ]

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.ScatterMultiple(
        datapaths,
        plotname=relname,
        titles=titles,
        format="png",
        alpha=0.5,
        scalebars=0.3,
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
