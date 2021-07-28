#!/usr/bin/env python
import jnb_msc

import shutil
import sys
import inspect
import string

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca")

    tsne = dsrc / "affinity/spectral/stdscale;f:1e-4/tsne;early_exaggeration:1"

    dpaths = [
        tsne
    ]
    titles = [
        ""
    ]

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.ScatterSingle(
        dpaths,
        plotname=relname,
        titles=titles,
        format="pdf",
        lettering=False,
        scalebars=0.25,
        # figwidth=2.25,
        # figheight=1.45,
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

    fig.savefig(sys.argv[3], format="pdf", dpi=300)
