#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/tasic")

    lo_exag = 4
    hi_exag = 30

    datafiles, titles = jnb_msc.plot.SixPanelPlot.panel_datapaths(
        dsrc, lo_exag=lo_exag, hi_exag=hi_exag
    )
    corrs_f = Path() / "../../stats/tasic_corr.csv"

    datafiles, titles = jnb_msc.plot.SixPanelPlot.panel_datapaths(
        dsrc, lo_exag=lo_exag, hi_exag=hi_exag
    )

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = Path(sys.argv[2])
    plotter = jnb_msc.plot.ExtPanelPlot(
        datafiles,
        corrs_f.absolute(),
        plotname=relname,
        titles=titles,
        format=relname.suffix.replace(".", ""),
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

    redo.redo_ifchange(list(filedeps) + datadeps + [corrs_f])
    plotter.load()
    shape = plotter.data[0].shape
    plotter.data[0] += plotter.random_state.normal(0, 1e-4, size=shape)
    fig, axs = plotter.transform()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
