#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    data = "zfish"
    dsrc = Path(f"../../data/{data}")

    lo_exag = 4
    hi_exag = 30

    datafiles, titles = jnb_msc.plot.SixPanelPlot.panel_datapaths(
        dsrc, lo_exag=lo_exag, hi_exag=hi_exag, k=150
    )
    # corrs_f = Path() / f"../../stats/{data}_corr.csv"

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.SixPanelPlot(
        datafiles,
        plotname=relname,
        titles=titles,
        format="png",
        lo_exag=lo_exag,
        hi_exag=hi_exag,
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
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
