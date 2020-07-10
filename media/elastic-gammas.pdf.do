#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect

# from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def format_gamma(gamma):
    return f"\\frac{{1}}{{{gamma}}}" if gamma > 1 else f"{gamma}"


def learning_rate(gamma):
    return min(1, gamma / 1000)


if __name__ == "__main__":
    dsrc = Path("../data/mnist/pca/ann/rownorm/stdscale;f:1e-4")

    datapaths = []
    titles = []

    gamma1_iter = 5000
    eps = 0.001

    for gamma in [10000, 100, 1]:
        lr = learning_rate(gamma)

        datapaths.append(
            dsrc
            / (
                f"tsnee;early_exaggeration:1;elastic_const:{gamma};learning_rate:{lr};eps:{eps}"
                + (f";n_iter:{gamma1_iter}" if gamma == 1 else "")
            )
        )

        g = 1 / gamma  # format_gamma(gamma)
        titles.append(f"$\\gamma=\\mathdefault{{{g:g}}}$")

    prep = "tsnee;early_exaggeration:1;elastic_const:10000;learning_rate:1;eps:1/"
    for gamma in [100, 1]:
        lr = learning_rate(gamma)

        datapaths.append(
            dsrc
            / prep
            / (
                f"tsnee;learning_rate:1;early_exaggeration:1;elastic_const:{gamma};learning_rate:{lr};eps:{eps}"
                + (f";n_iter:{gamma1_iter}" if gamma == 1 else "")
            )
        )

        g = 1 / gamma
        titles.append(f"$\\gamma=\\mathdefault{{{g:g}}}$, EE")

    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotRow(
        datapaths,
        plotname=relname,
        titles=titles,
        format="pdf",
        scalebars=0.3,
        figheight=1.1,  # 1.25
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
