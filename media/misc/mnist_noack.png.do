#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect
import string

# from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca/ann")

    n_iter = 1000

    datapaths = []
    titles = []

    for r in range(0, -3, -1):
        for a in range(0, 3):
            title = f"$({a}, {r})$\n"
            if a == 2 and r == -1:
                title = "FR, " + title
            elif a == 1 and r == -1:
                title = "FA2, " + title
            elif a == 0 and r == -1:
                title = "LL, " + title

            # if (a == 0 and r == -1) or (a == 1 and r == 0):
            #     std = 1000
            if a - r == 1:
                std = 1000
                n_iter = 2000
            else:
                std = 10
                n_iter = 1000

            src = dsrc / f"stdscale;f:{std}"

            title += f", $std={std}$" if std != 10 else ""

            if a != 1 or r != -1:
                run = f"noack;a:{a};r:{r};n_iter:{n_iter}"
            else:
                run = f"noack;n_iter:{n_iter}"

            # due to using adam, the lr is also an upper bound for the gradient
            if (a == 0) or (r == 0):
                lr = 10
                run += ";max_step_norm:100"
            else:
                lr = 0.1  # default
            run += f";learning_rate:{lr}" if lr != 0.1 else ""
            title += f", $\\eta={lr}$" if lr != 0.1 else ""

            if r < -1:
                eps = 0.1
            else:
                eps = 0
            title += f", $\\epsilon={eps}$" if eps != 0 else ""
            run += f";eps:{eps}" if eps != 0 else ""

            datapaths.append(src / run)
            titles.append(title)

    relname = sys.argv[2]
    plotter = jnb_msc.plot.ScatterMultiple(
        datapaths,
        plotname=relname,
        titles=titles,
        format="png",
        scalebars=0.3,
        alpha=1,
        dpi=600,
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
    axs.flat[0].remove()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
