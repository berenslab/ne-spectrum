#!/usr/bin/env python
import jnb_msc

import os
import sys
import inspect
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib import colors as mplclrs
from pathlib import Path


def get_rhos():
    rhos = np.logspace(np.log10(1), np.log10(100)).round(1)
    return sorted(list(rhos) + [4, 30])


def tsne_from_rho(rho, dsrc):
    tsne = "tsne"
    if rho > 12:
        tsne += f";early_exaggeration:{rho:g}"

    if rho != 1:
        tsne += f";late_exaggeration:{rho:g}"

    return dsrc / tsne / "data.npy"


def rescale(ar):
    """Rescale the given numpy array to fit into the range [0,1]."""
    return (ar - ar.min()) / (ar.max() - ar.min())


def animate_exaggeration(
    flist, rhos, labels, alpha=0.3, suptitle=None, fps=25, frac=0.25
):
    n = len(rhos)
    rows = 1
    cols = 2
    gs_kw = {"width_ratios": [0.9, 0.1]}
    fig, axs = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(2 * 1.1, 2),
        dpi=300,
        gridspec_kw=gs_kw,
        constrained_layout=True,
    )
    ax_sc = axs[0]
    ax_bar = axs[1]

    data = np.load(flist[0])
    d = rescale(data)
    sc = ax_sc.scatter(d[:, 0], d[:, 1], c=labels, alpha=alpha)
    bar = ax_bar.bar(0, max(rhos), log=True)
    ax_bar.set_ylim(min(rhos), max(rhos))

    ax_sc.set_axis_off()

    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.yaxis.tick_right()
    ax_bar.yaxis.set_ticks([])
    ax_bar.yaxis.set_ticks([], minor=True)
    ticks = [1, 4, 30, 100]
    ax_bar.yaxis.set_ticks(ticks)
    ax_bar.yaxis.set_ticklabels([str(t) for t in ticks])
    ax_bar.set_xticks([])
    ax_bar.tick_params(labelsize="small")

    exag_title = ax_bar.set_title(
        f"{rhos[0]:.1f}$=\\rho$", loc="right", fontdict=dict(fontsize="medium")
    )

    scalebar = jnb_msc.plot.add_scalebar_frac(ax_sc, frac_len=frac)
    round10 = lambda x: 10 ** int(np.floor(np.log10(2 * x)))

    if suptitle is not None:
        fig.suptitle(suptitle)

    fig.set_constrained_layout(False)

    def update(i, scatter, bar, exagtxt, scalebar):
        rho = rhos[i]
        f = flist[i]

        data = np.load(f)
        d = rescale(data)
        scatter.set_offsets(d)
        bar[0].set_height(rho)

        exagtxt.set_text(f"{rho:.1f}$=\\rho$")

        l = frac * (data.max() - data.min())
        l10 = round10(l)
        ## deconstruct the scalebar object.  Look at the source to
        ## make sense of what happens here.
        vpack = scalebar.get_children()[0]
        auxtrbox, txtarea = vpack.get_children()
        arrow = auxtrbox.get_children()[0]
        arrow.set_positions((0, 0), (l10 / data.max(), 0))
        txtarea.set_text(f"{l10:g}")

        return scatter, bar[0], exagtxt, scalebar

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n,
        fargs=(sc, bar, exag_title, scalebar),
        interval=fps,
        blit=True,
        save_count=n,
        repeat=False,
    )

    return anim


if __name__ == "__main__":
    dsrc = Path("../../data/mnist")
    prefix = dsrc / "pca/affinity/stdscale;f:1e-4"

    fps = 25
    meta = {"artist": "Niklas Böhm"}

    rhos = list(np.arange(0.1, 1, 0.1).round(1)) + get_rhos()
    rhos = np.array(rhos)

    small = rhos[rhos < 1]
    lower_rhos = list(np.hstack([small[::-1], [min(small)] * (fps // 4), small]))
    rho1 = list(rhos[rhos == 1])
    rhos = (
        rho1 * (fps // 2)
        + lower_rhos
        + rho1 * (fps // 4)
        + list(rhos[rhos > 1])
        + list(rhos[rhos > 1][::-1])
    )

    # rhos = [rhos[0]] * (fps // 2) + rhos + [rhos[-1]] * (fps // 2)
    flist = []

    for rho in rhos:
        fname = tsne_from_rho(rho, prefix)
        flist.append(fname)

    relname = sys.argv[2]
    plotter = jnb_msc.plot.ScatterSingle(flist[0].parent)
    rc = plotter.rc
    labelname = dsrc / "labels.npy"

    jnb_msc.redo.redo_ifchange(flist + [rc, labelname])
    labels = np.load(labelname)

    with plt.rc_context(fname=rc):
        anim = animate_exaggeration(flist, rhos, labels, fps=fps)

        anim.save(
            sys.argv[3], fps=fps, metadata=meta, extra_args=["-f", "mp4"],
        )

    # with tempfile.NamedTemporaryFile("w+") as f:
    #     [f.write(str(fn) + "\n") for fn in flist]
    #     f.flush()
    #     fn = Path(f.name)
    #     plotter = jnb_msc.anim.ScatterAnimations(
    #         [fn.parent],
    #         dataname=fn.name,
    #         labelname=(dsrc / "labels.npy").absolute(),
    #         plotname=relname,
    #         format="mp4",
    #         scalebars=0.3,
    #         fps=25,
    #     )
    #     filedeps = set(
    #         [
    #             mod.__file__
    #             for mod in [inspect.getmodule(m) for m in plotter.__class__.mro()]
    #             if hasattr(
    #                 mod, "__file__"
    #             )  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
    #         ]
    #     )

    #     jnb_msc.redo.redo_ifchange(list(filedeps) + flist)
    #     plotter()
    # # link to the result
    # os.link(plotter.outdir / relname, sys.argv[3])
