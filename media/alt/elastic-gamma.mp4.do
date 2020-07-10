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
    dsrc = Path("../../data/mnist/pca/ann/rownorm/stdscale;f:1e-4")
    prep = "tsnee;learning_rate:1;early_exaggeration:1;elastic_const:10000;eps:1/"

    datapaths = []
    titles = []

    for gamma in [100, 50, 10, 1]:
        lr = (1 / 10) / (100 / gamma)

        d = (
            dsrc
            / prep
            / f"tsnee;learning_rate:1;early_exaggeration:1;elastic_const:{gamma};learning_rate:{lr};eps:1;save_iter_freq:1;n_iter:1250"
        )
        title = f"$\\eta=${lr:g}, $\\gamma=\\frac{{1}}{{{gamma}}}$"

        titles.append(title)
        datapaths.append(d)

    relname = sys.argv[2]
    plotter = jnb_msc.anim.ScatterAnimations(
        datapaths, plotname=relname, titles=titles, format="mp4", scalebars=0.3
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
    plotter()
    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
