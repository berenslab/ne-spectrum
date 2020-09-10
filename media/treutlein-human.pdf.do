#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../data/treutlein_409b2")

    datafiles, titles = jnb_msc.plot.SixPanelPlot.panel_datapaths(dsrc)
    corr_f = Path("../stats/thuman_corr.csv")

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.SixPanelPlotsExt(
        datafiles,
        corr_f.absolute(),
        plotname=relname,
        titles=titles,
        format="pdf",
        lettering=True,
        figwidth=1.625,
    )
    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in plotter.__class__.mro()]
            if hasattr(mod, "__file__")  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
        ]
    )

    datadeps = plotter.get_datadeps()

    redo.redo_ifchange(list(filedeps) + datadeps + [corr_f])
    plotter.load()
    plotter.data[0][:, 1] *= -1
    plotter.transform()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
