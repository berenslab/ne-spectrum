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
            title = f"$({a}, {r})$, " + r"\\"
            if a == 2 and r == -1:
                title = r"\gls{fr}, " + title
            elif a == 1 and r == -1:
                title = r"\gls{fa2}, " + title
            elif a == 0 and r == -1:
                title = r"\textsc{ll}, " + title

            # if (a == 0 and r == -1) or (a == 1 and r == 0):
            #     std = 1000
            if a == 1 and r == 0:
                std = 10000
                n_iter = 750
            elif a == 0 and r == -1:
                std = 100
                n_iter = 2000
            elif a == 0 and r == -2:
                std = 100
                n_iter = 2000
            elif a == 2 and r == 0:
                std = 100
                n_iter = 750
            else:
                std = 10
                n_iter = 750

            src = dsrc / f"stdscale;f:{std}"
            title += f"$\\sigma={std}$, "

            if a != 1 or r != -1:
                run = f"noack;a:{a};r:{r};n_iter:{n_iter};save_iter_freq:1"
            else:
                run = f"noack;n_iter:{n_iter};save_iter_freq:1"

            # due to using adam, the lr is also an upper bound for the gradient
            if a - r > 2:
                lr = 0.01
            elif a == 0 and r == -2:
                lr = 1
            elif a == 0 and r == -1:
                lr = 10
                run += ";max_step_norm:100"
            elif a == 1 and r == 0:
                lr = 100
                run += ";max_step_norm:1000"
            else:
                lr = 0.1  # default
            run += f";learning_rate:{lr}" if lr != 0.1 else ""
            title += f"$\\eta={lr}$, "

            if a > 0 and r == -2:
                eps = 0.1
            else:
                eps = 0
            title += f"$\\epsilon={eps}$, " if eps != 0 else ""
            run += f";eps:{eps}" if eps != 0 else ""

            if title.endswith(", "):
                title = title[: -len(", ")]

            datapaths.append(src / run)
            titles.append(title)

    # Replace the pathological case a=r=0 with something that has been
    # computed already.  Will be removed anyways
    datapaths = [datapaths[1]] + datapaths[1:]
    letters = "x" + string.ascii_lowercase

    relname = Path(sys.argv[2])
    plotter = jnb_msc.plot.PlotMultWithTitle(
        datapaths,
        plotname=relname,
        titles=titles,
        format=relname.suffix.replace(".", ""),
        scalebars=0.3,
        lettering=letters,
        alpha=1,
        figwidth=1.5
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
    figs = plotter.transform()
    with plt.rc_context(fname=plotter.rc):
        figs[0].get_axes()[0].remove()
        # tprops = {"size": "x-large"}
        tprops = {"usetex": True}
        plotter.add_inset_legend(
            figs[4].get_axes()[0], plotter.data[4], plotter.labels,
            textprops=tprops,
        )
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
