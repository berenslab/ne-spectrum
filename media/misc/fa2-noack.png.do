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
    dsrc = Path("../../data/mnist/pca/ann")

    datapaths = [dsrc / "fa2;use_degrees:0", dsrc / "noack"]
    titles = ["FA2 no rep. by degrees", "Noack $a=1$, $r=-1$"]

    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotRow(
        datapaths,
        plotname=relname,
        titles=titles,
        format="png",
        scalebars=0.3,
        alpha=1,
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
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
