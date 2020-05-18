#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

from pathlib import Path


if __name__ == "__main__":
    ps = []
    for i in [-6]:
        ps.append(f"../data/cchains/pca/ann/noack;r:{i}")

    relname = sys.argv[2]
    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    plotter = jnb_msc.plot.ScatterMultiple(ps, plotname=relname)
    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in plotter.__class__.mro()]
            if hasattr(mod, "__file__")  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
        ]
    )

    datadeps = plotter.get_datadeps()

    redo.redo_ifchange(list(filedeps) + datadeps)
    plotter()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
