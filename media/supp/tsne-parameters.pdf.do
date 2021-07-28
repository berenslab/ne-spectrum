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
    prefix = dsrc / "affinity"

    std = 50
    setups = [
        "stdscale;f:1e-4/tsne",
        "stdscale;f:1e-4/tsne;early_exaggeration:1/",
        f"stdscale;f:{std:g}/tsne",
    ]

    datafiles = [
        prefix / r / tsne for r in [".", "random"] for tsne in setups
    ]

    titles = [
        "Early exaggeration,\nInitial std${}={}$0.0001,\nPCA init",
        "No early exaggeration,\nInitial std${}={}$0.0001,\nPCA init",
        f"Early exaggeration,\nInitial std${{}}={{}}${std},\nPCA init",
        "Early exaggeration,\nInitial std${}={}$0.0001,\nrandom init",
        "No early exaggeration,\nInitial std${}={}$0.0001,\nrandom init",
        f"Early exaggeration,\nInitial std${{}}={{}}${std},\nrandom init",
    ]

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.ScatterMultiple(
        datafiles, plotname=relname, titles=titles, format="pdf", scalebars=0.2, layout=(2,3),
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
    fig.savefig(sys.argv[3], format="pdf", dpi=200)
