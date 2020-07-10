#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect
import tempfile
import shutil

# from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == "__main__":
    dsrc = Path("../../data/mnist/pca/ann")

    n_iter = 750
    n_iters = []

    datapaths = []
    titles = []

    for r in range(0, -3, -1):
        for a in range(0, 3):
            title = f"({a}, {r})\n"
            if a == 2 and r == -1:
                title = "FR, " + title
            elif a == 1 and r == -1:
                title = "FA2, " + title
            elif a == 0 and r == -1:
                title = "LL, " + title

            # if (a == 0 and r == -1) or (a == 1 and r == 0):
            #     std = 1000
            if a == 1 and r == 0:
                std = 10000
                n_iter = 750
            elif a == 0 and r == -1:
                std = 100
                n_iter = 2000
            elif a == 0 and r == -2:
                std = 100
                n_iter = 2000
            elif a == 2 and r == 0:
                std = 100
                n_iter = 750
            else:
                std = 10
                n_iter = 750

            src = dsrc / f"stdscale;f:{std}"
            title += f"$\\sigma=${std}, "

            if a != 1 or r != -1:
                run = f"noack;a:{a};r:{r};n_iter:{n_iter};save_iter_freq:1"
            else:
                run = f"noack;n_iter:{n_iter};save_iter_freq:1"

            # due to using adam, the lr is also an upper bound for the gradient
            if a - r > 2:
                lr = 0.01
            elif a == 0 and r == -2:
                lr = 1
            elif a == 0 and r == -1:
                lr = 10
                run += ";max_step_norm:100"
            elif a == 1 and r == 0:
                lr = 100
                run += ";max_step_norm:1000"
            else:
                lr = 0.1  # default
            run += f";learning_rate:{lr}" if lr != 0.1 else ""
            title += f"$\\eta=${lr}, "

            if a > 0 and r == -2:
                eps = 0.1
            else:
                eps = 0
            title += f"$\\epsilon=${eps}, " if eps != 0 else ""
            run += f";eps:{eps}" if eps != 0 else ""

            if title.endswith(", "):
                title = title[: -len(", ")]

            datapaths.append(src / run)
            titles.append(title)
            n_iters.append(n_iter)

    relname = sys.argv[2]

    dummy_anim = jnb_msc.anim.ScatterAnimations(datapaths)
    # save a png of the final frame (not tracked by redo)
    plotter2 = jnb_msc.plot.ScatterMultiple(
        datapaths,
        plotname=Path(relname).with_suffix(".mp4.png").absolute(),
        titles=titles,
        format="png",
        scalebars=0.3,
        alpha=1,
        dpi=200,
    )
    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in dummy_anim.__class__.mro()]
            if hasattr(mod, "__file__")  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
        ]
    )

    jnb_msc.redo.redo_ifchange(
        list(filedeps) + dummy_anim.get_datadeps() + plotter2.get_datadeps()
    )
    dummy_anim.load()
    tmpdirs = []

    for i, (t, dp, n_iter) in enumerate(zip(titles, datapaths, n_iters)):
        tmpd = Path(tempfile.mkdtemp())
        tmpdirs.append(tmpd)

        # extend the current flist so that all have equal length.
        flist = dummy_anim.dataffiles[i]
        flist += [dummy_anim.dataffiles[i][-1]] * (max(n_iters) - n_iter)

        with open(tmpd / "data.flist", "w") as f:
            [f.write(str(fn) + "\n") for fn in flist]

    plotter = jnb_msc.anim.ScatterAnimations(
        tmpdirs,
        labelname=(dsrc / "labels.npy").absolute(),
        titles=titles,
        # use the proper output since the dirnames are messed up
        plotname=Path(sys.argv[3]).absolute(),
        format="mp4",
        dpi=100,
    )
    plotter2.load()
    fig, axs = plotter2.transform()
    axs.flat[0].remove()
    plotter2.save()
    plotter()

    for tmpd in tmpdirs:
        shutil.rmtree(tmpd)
