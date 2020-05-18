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
    fa2 = dsrc / ann_prefix / "stdscale;f:1e4" / init / "fa2;n_iter:5000"
    umap = dsrc / umap_prefix / "maxscale;f:10" / init / "umap"

    tsne_prefix = "stdscale;f:1e-4/affinity"
    if perplexity is not None:
        tsne_prefix += f";perplexity:{perplexity}"
    elif k is not None:
        tsne_prefix += f";perplexity:{k_ratio*k}"

    tsne = dsrc / tsne_prefix / init / "tsne/"
    tsne4 = dsrc / tsne_prefix / init / f"tsne;late_exaggeration:{lo_exag}"
    tsne30 = (
        dsrc
        / tsne_prefix
        / init
        / f"tsne;early_exaggeration:{hi_exag};late_exaggeration:{hi_exag}"
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
    dsrc = Path("../data/cchains")
    init_name = "pca"

    datafiles, titles = panel_datapaths(  # jnb_msc.plot.SixPanelPlot.
        dsrc, hi_exag=30, lo_exag=2, init=init_name
    )

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.SixPanelPlot(
        datafiles, plotname=relname, titles=titles, format="png"
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
    import matplotlib.pyplot as plt

    with plt.rc_context(fname=plotter.rc):
        # init = np.load(dsrc / "random/data.npy")
        init = np.load(datafiles[-1].parent / "data.npy")
        ax = [ax for ax in plotter.fig.get_axes() if ax.get_label() == "legend"].pop()
        ax.scatter(init[:, 0], init[:, 1], c=plotter.labels, alpha=plotter.alpha)
        ax.set_title(init_name + " (init)")

        xmin, ymin = init.min(axis=0)[:2]
        xmax, ymax = init.max(axis=0)[:2]
        ax.axis([xmin, xmax, ymin, ymax])
        jnb_msc.plot.set_aspect_center(ax)

    plotter.save()

    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
