#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np

from pathlib import Path


def panel_datapaths(
    datasource, k=None, perplexity=None, k_ratio=2, lo_exag=4, hi_exag=30, init=".",
):
    """Return a pair of 6-tuples suitable for this class' plotting
    regime.

    Returns
    -------
    A pair of data filenames and titles."""
    dsrc = Path(datasource)

    ann_prefix = "ann" + ("" if k is None else f";n_neighbors:{k}")
    umap_prefix = "umap_knn" + ("" if k is None else f";n_neighbors:{k}")

    spectral = dsrc / ann_prefix / "spectral"
    fa2 = dsrc / ann_prefix / init / "stdscale;f:10" / "fa2"
    umap = dsrc / umap_prefix / init / "maxscale;f:10" / "umap"

    tsne_prefix = "affinity"
    if perplexity is not None:
        tsne_prefix += f";perplexity:{perplexity}"
    elif k is not None:
        tsne_prefix += f";perplexity:{k_ratio*k}"

    tsne = dsrc / tsne_prefix / init / "stdscale;f:1e-4/tsnestage/"
    tsne4 = (
        dsrc / tsne_prefix / init / f"stdscale;f:1e-4/tsnestage;exaggeration:{lo_exag}"
    )
    tsne30 = (
        dsrc / tsne_prefix / init / f"stdscale;f:1e-4/tsnestage;exaggeration:{hi_exag}"
    )
    tsnes = [tsne, tsne4, tsne30]

    titles = [
        "Laplacian Eigenmaps",
        "ForceAtlas2",
        "UMAP",
        "t-SNE",
        # hack around to use the regular font for the numbers.
        # This uses dejavu font for \rho, which sucks, but is
        # hardly noticeable (fortunately).  The alternative would
        # be to use the upright version of the correct font by
        # specifying \mathdefault{\rho}
        f"$\\rho = {{}}${lo_exag}",
        f"$\\rho = {{}}${hi_exag}",
    ]

    return [spectral, fa2, umap] + tsnes, titles


if __name__ == "__main__":
    dsrc = Path("../data/gauss_devel")
    init_name = "random"

    datafiles, titles = panel_datapaths(  # jnb_msc.plot.SixPanelPlot.
        dsrc, hi_exag=30, lo_exag=2, init=init_name
    )

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.SixPanelPlot(
        datafiles, plotname=relname, titles=titles, format="pdf", alpha=0.5
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
    plotter.transform()
    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
