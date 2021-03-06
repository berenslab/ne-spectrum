#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as mfm

from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/kuzmnist/pca")

    lo_exag = 4
    hi_exag = 30

    datafiles, titles = jnb_msc.plot.SixPanelPlot.panel_datapaths(
        dsrc, lo_exag=lo_exag, hi_exag=hi_exag
    )
    trans_f = Path() / "../../static/df_unicode_sym.csv"
    font_path = "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf"
    corrs_f = Path() / "../../stats/kuzmnist_corr.csv"

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

    redo.redo_ifchange(list(filedeps) + datadeps + [trans_f, corrs_f])
    df = pd.read_csv(trans_f)

    plotter.load()
    plotter.data[0] *= -1
    fig, axs = plotter.transform()

    with plt.rc_context(fname=plotter.rc):
        tr = lambda n: df["glyph"][
            (df["language"] == "Hiragana") & (df["num"] == n)
        ].item()
        prop = mfm.FontProperties(fname=font_path)
        plotter.add_inset_legend(
            axs[0, 3],
            plotter.data[3],
            plotter.labels,
            to_str=tr,
            posfun="kde",
            textprops={"font_properties": prop, "fontsize": "x-large"},
        )

    plotter.save()
    # link to the result
    os.link(plotter.outdir / relname, sys.argv[3])
