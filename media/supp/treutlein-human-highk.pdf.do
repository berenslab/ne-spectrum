#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/treutlein_409b2")

    datafiles, titles = jnb_msc.plot.SixPanelPlot.panel_datapaths(
        dsrc, k=150, lo_exag=3, hi_exag=20
    )
    spectral, fa2, umap, tsne, tsne4, tsne30 = datafiles
    datafiles = [spectral, tsne30, tsne4, tsne, fa2, umap]

    spectral, fa2, umap, tsne, tsne4, tsne30 = titles
    titles = [spectral, tsne30, tsne4, tsne, fa2, umap]


    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotMultWithTitle(
        datafiles,
        plotname=relname,
        titles=titles,
        format="pdf",
        figwidth=1.5,
        scalebars=0.25,
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
    plotter.data[0][:, 0] *= -1
    plotter.transform()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
