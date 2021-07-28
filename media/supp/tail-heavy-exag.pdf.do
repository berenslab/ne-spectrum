#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca")

    tsne_default = dsrc / "affinity/stdscale;f:1e-4/tsne"

    dpaths = []
    titles = []

    dofs = [100, 1, 0.5]
    exags = [50, 4, 1, 0.5]
    for dof in dofs:
        for exag in exags:
            opts = ""
            if dof != 1:
                opts += f";dof:{dof:g}"
            if exag > 12:
                opts += f";early_exaggeration:{exag:g}"
            if exag != 1:
                opts += f";late_exaggeration:{exag:g}"

            if dof == exag:
                title = f"$\\alpha=\\rho={dof}$"
            else:
                title = f"$\\alpha={dof}$, $\\rho={exag}$"

            tsne = tsne_default.parent / (tsne_default.name + opts)
            dpaths.append(tsne)
            titles.append(title)

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.ScatterMultiple(
        dpaths,
        plotname=relname,
        titles=titles,
        format="pdf",
        scalebars=0.25,
        layout=(len(dofs), len(exags)),
        # figheight=3,
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

    fig.savefig(sys.argv[3], format="pdf", dpi=200)
