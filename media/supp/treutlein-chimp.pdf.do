#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/treutlein")

    datafiles, titles = jnb_msc.plot.SixPanelPlot.panel_datapaths(
        dsrc, lo_exag=7, hi_exag=40
    )
    corr_f = Path('../../stats/tchimp_corr.csv')

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.SixPanelPlotsExt(
        datafiles,
        corr_f.absolute(),
        plotname=relname,
        titles=titles,
        format="pdf",
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
    # fix spectral embedding to be consistent with the other scatters
    plotter.data[0] *= -1
    # flip all plots around their x to align with the human dataset
    for i in range(len(plotter.data)):
        plotter.data[i][:, 0] *= -1
    plotter.transform()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
