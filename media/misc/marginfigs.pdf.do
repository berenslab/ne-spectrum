#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect
import string

from umap.umap_ import find_ab_params
import numpy as np

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca")
    knn_prefix = "ann/spnorm"

    tsne_default = dsrc / "affinity/stdscale;f:1e-4/tsne"
    tsne_knn = dsrc / knn_prefix / "stdscale;f:1e-4/tsne"

    min_dist = 0.1
    spread = 1.0
    a, b = find_ab_params(spread, min_dist)

    umap = dsrc / "umap_knn/maxscale;f:10/umap"
    umap_eps = dsrc / "umap_knn/maxscale;f:10/umap;eps:1"
    fa2_degrep = dsrc / "ann/fa2"
    tsne_no_ee = dsrc / "affinity/stdscale;f:1e-4/tsne;early_exaggeration:1/"
    tsne_stdscale = dsrc / "affinity/stdscale;f:25/tsne"

    dpaths = [
        tsne_default,
        umap,
        umap_eps,
        fa2_degrep,
        tsne_no_ee,
        tsne_stdscale,
    ]
    titles = [
        "\gls{tsne}",
        "\gls{umap}",
        "\gls{umap}, $\epsilon=1$",
        "\gls{fa2}",
        "\gls{tsne}, no early exaggeration",
        "\gls{tsne}, initial $\sigma=25$"
    ]

    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    relname = sys.argv[2]
    plotter = jnb_msc.plot.PlotMultWithTitle(
        dpaths,
        plotname=relname,
        titles=titles,
        format="pdf",
        scalebars=0.20,
        figwidth=2,
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
    # flip the default UMAP with LE init
    plotter.data[4][:, 1] *= -1
    plotter.data[-2] *= -1
    figs = plotter.transform()
    #

    # with plt.rc_context(fname=plotter.rc):
    #     gs = axs[0, 0].get_gridspec()
    #     ax = fig.add_subplot(gs[1, :])
    #     ax.set_title("UMAP\n")
    #     ax.set_axis_off()
    plotter.save()

    # link to the result
    # os.link(plotter.outdir / relname, sys.argv[3])
