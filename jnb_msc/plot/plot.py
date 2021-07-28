from ..abpc import ProjectBase
from ..util import name_to_class, name_and_dict
from .scalebars import add_scalebar_frac
from .special import plot_correlation_rhos

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

import string
import inspect
import os


class ScatterMultiple(ProjectBase):
    """Takes a list of paths and creates a scatter plot from the
    information."""

    def __init__(
        self,
        paths,
        dataname=None,
        labelname=None,
        plotname=None,
        titles=None,
        lim_eps=0.025,
        alpha=0.5,
        format="png",
        rc=None,
        scalebars=None,
        lettering=True,
        figwidth=5.5,  # in inches
        figheight=None,
        layout=None,
        **kwargs,
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        path = Path(os.path.commonpath(self.paths))
        super().__init__(
            path, indir=path.absolute(), outdir=path.absolute(), random_state=None
        )
        self.dataname = "data.npy" if dataname is None else dataname
        self.labelname = self.resolve_path(self.path, labelname, "labels.npy")
        self.plotname = self.resolve_path(self.path, plotname, "data.png")
        self.titles = titles

        self.lim_eps = lim_eps
        self.alpha = alpha

        self.format = format
        self.scalebars = scalebars
        self.lettering = lettering
        self.kwargs = kwargs
        self.figwidth = figwidth
        self.figheight = figheight
        self.layout = layout

        if rc is None:
            # find the project default file, located in the same dir
            # where the code for this class is located.
            modfile = Path(inspect.getmodule(type(self)).__file__)
            self.rc = modfile.parent / "matplotlibrc"
        else:
            self.rc = rc

    def get_datadeps(self):
        return [self.rc, self.labelname, *[p / self.dataname for p in self.paths]]

    def load(self):
        self.labels, *self.data = [np.load(f) for f in self.get_datadeps()[1:]]

    def transform(self):
        if self.titles is None:
            titles = titles_from_paths(self.paths)
        else:
            titles = self.titles

        if self.layout is None:
            rows, cols = auto_layout(len(self.data))
        else:
            rows, cols = self.layout

        if self.figheight is None:
            width = (self.figwidth / cols) * rows
        else:
            width = self.figheight

        try:
            letter_iter = iter(self.lettering)
        except TypeError:
            letter_iter = string.ascii_lowercase

        with plt.rc_context(fname=self.rc):
            fig, axs = plt.subplots(
                nrows=rows,
                ncols=cols,
                figsize=(self.figwidth, width),
                squeeze=False,
                constrained_layout=True,
                **self.kwargs,
            )

            ax_iter = iter(axs.flat)
            for ax, dat, title, letter in zip(ax_iter, self.data, titles, letter_iter):
                ax.scatter(
                    dat[:, 0],
                    dat[:, 1],
                    c=self.labels,
                    alpha=self.alpha,
                    rasterized=True,
                )

                xmin = dat[:, 0].min()
                xmax = dat[:, 0].max()

                ymin = dat[:, 1].min()
                ymax = dat[:, 1].max()

                lim_min = lambda x: x * (1 - np.sign(x) * self.lim_eps)
                lim_max = lambda x: x * (1 + np.sign(x) * self.lim_eps)
                ax.set_xlim(lim_min(xmin), lim_max(xmax))
                ax.set_ylim(lim_min(ymin), lim_max(ymax))

                if title != "":
                    ax.set_title(title)

                set_aspect_center(ax)
                if self.scalebars:
                    self.add_scalebar(ax, self.scalebars)

                if self.lettering and len(ax_iter) > 1:
                    self.add_lettering(ax, letter)

            # if any axes are left empty, remove them
            for ax in ax_iter:
                ax.remove()

        self.data_ = fig, axs
        self.fig = fig
        self.axs = axs
        return self.data_

    def save(self):
        save = lambda f, data: data.savefig(f, format=self.format)
        self.save_lambda(self.plotname, self.fig, save)

    @staticmethod
    def add_inset_legend(ax, data, labels, to_str=str, textprops=None, posfun="mean"):
        for i in np.unique(labels):
            mask = labels == i
            box = TextArea(to_str(i), textprops=textprops)

            if posfun == "mean":
                pos = data[mask].mean(axis=0)
            elif posfun == "median":
                pos = np.median(data[mask], axis=0)
            elif posfun == "kde":
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(data[mask].T)
                pos = data[mask][kde(data[mask].T).argmax()]
            abox = AnnotationBbox(box, pos, frameon=False, pad=0)
            ax.add_artist(abox)

    @staticmethod
    def add_scalebar(ax, frac_len=0.125):
        add_scalebar_frac(ax, frac_len=frac_len)

    @classmethod
    def add_lettering(cls, ax, letter, fontdict=None, loc="left", **kwargs):
        if fontdict is None:
            fontdict = cls.get_letterdict()

        other_title = ax.get_title("center")
        newlines = len(other_title.split("\n")) - 1
        ax.set_title(letter + newlines * "\n", fontdict=fontdict, loc=loc, **kwargs)

    @staticmethod
    def get_letterdict():
        return {
            "fontsize": "x-large",
            "fontweight": "bold",
        }


class ScatterSingle(ScatterMultiple):
    """Transform a NumPy data file (.npy) into a scatter plot."""

    def __init__(
        self,
        paths,
        dataname=None,
        labelname=None,
        plotname=None,
        titles=None,
        lim_eps=0.025,
        alpha=0.5,
        format="png",
        rc=None,
        scalebars=None,
        **kwargs,
    ):
        super().__init__(
            paths,
            dataname=dataname,
            labelname=labelname,
            plotname=plotname,
            titles=titles,
            lim_eps=lim_eps,
            alpha=alpha,
            format=format,
            rc=rc,
            scalebars=scalebars,
            **kwargs,
        )
        self.figwidth = 2


class SixPanelPlot(ScatterMultiple):
    def __init__(
        self,
        paths,
        dataname=None,
        labelname=None,
        plotname=None,
        titles=None,
        lim_eps=0.025,
        alpha=0.5,
        format="png",
        rc=None,
        scalebars=0.25,
        **kwargs,
    ):
        super().__init__(
            paths,
            dataname=dataname,
            labelname=labelname,
            plotname=plotname,
            titles=titles,
            lim_eps=lim_eps,
            alpha=alpha,
            format=format,
            rc=rc,
            scalebars=scalebars,
            **kwargs,
        )

    def transform(self):
        spectral, fa2, umap, tsne, tsne4, tsne30 = self.data
        labels = self.labels
        alpha = self.alpha

        width_inch = 5.5  # text (and thus fig) width for nips
        rows = 2
        cols = 4
        box_inch = width_inch / cols
        with plt.rc_context(fname=self.rc):
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(width_inch, rows * box_inch),
                dpi=600,
                constrained_layout=True,
            )
            t_sp, t_fa, t_umap, *t_tsne = self.titles

            axs[0, 0].set_label("spectral")
            axs[0, 0].set_title(t_sp)
            axs[0, 0].scatter(
                spectral[:, 0], spectral[:, 1], c=labels, alpha=alpha, rasterized=True
            )
            axs[0, 3].set_title(t_tsne[0])
            axs[0, 3].scatter(
                tsne[:, 0], tsne[:, 1], c=labels, alpha=alpha, rasterized=True
            )

            axs[1, 1].set_title(t_fa)
            axs[1, 1].scatter(
                fa2[:, 0], fa2[:, 1], c=labels, alpha=alpha, rasterized=True
            )
            axs[1, 2].set_title(t_umap)
            axs[1, 2].scatter(
                umap[:, 0], umap[:, 1], c=labels, alpha=alpha, rasterized=True
            )

            # add t-sne with exaggeration
            exag_axs = []
            for title, data, g in zip(
                t_tsne[1:], [tsne4, tsne30], [axs[0, 2], axs[0, 1]]
            ):
                ax = fig.add_subplot(g, zorder=5)
                exag_axs.append(ax)
                ax.set_title(title)
                ax.scatter(
                    data[:, 0], data[:, 1], c=labels, alpha=alpha, rasterized=True
                )

            axs[1, 0].set_label("legend")
            axs[1, 3].remove()

            letter_iter = iter(string.ascii_lowercase)

            for ax in fig.get_axes():
                set_aspect_center(ax)
                if (
                    self.scalebars
                    and ax.get_label() != "spectral"
                    and ax.get_label() != "legend"
                ):
                    self.add_lettering(ax, next(letter_iter))
                    self.add_scalebar(ax, self.scalebars)
                elif ax.get_label() == "spectral":
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.set_frame_on(False)
                    self.add_lettering(ax, next(letter_iter))
                elif ax.get_label() == "legend":
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.set_frame_on(False)
                    # If we're plotting treutlein data, add the legend
                    # in the "legend" axes
                    if "treutlein" in [p[: len("treutlein")] for p in self.path.parts]:
                        add_treutlein_legend(ax, self.labels)
                    if "hydra" in [p[: len("hydra")] for p in self.path.parts]:
                        add_hydra_legend(ax, self.labels)
                    if "gauss_devel" in [
                        p[: len("gauss_devel")] for p in self.path.parts
                    ]:
                        add_gauss_devel_legend(ax, self.labels)
                else:
                    # Hide the right and top spines
                    ax.spines["right"].set_visible(False)
                    ax.spines["top"].set_visible(False)

                    ax.yaxis.set_ticks_position("left")
                    ax.xaxis.set_ticks_position("bottom")

        # fig.tight_layout()
        self.fig, self.axs = fig, axs
        self.data_ = fig, axs
        return fig, axs

    @staticmethod
    def panel_datapaths(
        datasource,
        k=None,
        perplexity=None,
        k_ratio=2,
        lo_exag=4,
        hi_exag=30,
        init=".",
        fa2_scale="stdscale;f:1e3",
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
        if fa2_scale is None:
            fa2 = dsrc / ann_prefix / init / "fa2"
        elif isinstance(fa2_scale, (int, float)):
            fa2 = dsrc / ann_prefix / init / f"stdscale;f:{fa2_scale}" / "fa2"
        else:
            fa2 = dsrc / ann_prefix / init / fa2_scale / "fa2"
        umap = dsrc / umap_prefix / init / "maxscale;f:10" / "umap"

        tsne_prefix = "affinity"
        if perplexity is not None:
            tsne_prefix += f";perplexity:{perplexity}"
        elif k is not None:
            tsne_prefix += f";perplexity:{k_ratio*k}"

        tsne = dsrc / tsne_prefix / init / "stdscale;f:1e-4" / "tsne/"
        tsne4 = (
            dsrc
            / tsne_prefix
            / init
            / "stdscale;f:1e-4"
            / f"tsne;late_exaggeration:{lo_exag}"
        )
        tsne30 = (
            dsrc
            / tsne_prefix
            / init
            / "stdscale;f:1e-4"
            / f"tsne;early_exaggeration:{hi_exag};late_exaggeration:{hi_exag}"
        )
        tsnes = [tsne, tsne4, tsne30]

        titles = [
            "Laplacian Eig.",
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


class PlotRow(ScatterMultiple):
    def transform(self):
        rows = 1
        cols = len(self.data)
        height = (self.figwidth / cols) if self.figheight is None else self.figheight

        if self.titles is None:
            titles = titles_from_paths(self.paths)
        else:
            titles = self.titles

        with plt.rc_context(fname=self.rc):
            fig, axs = plt.subplots(
                rows, cols, figsize=(self.figwidth, height), constrained_layout=True,
            )
            letter_dict = {
                "fontsize": "x-large",
                "fontweight": "bold",
            }
            for ax, d, title, letter in zip(
                axs, self.data, titles, string.ascii_lowercase
            ):
                ax.scatter(
                    d[:, 0], d[:, 1], c=self.labels, alpha=self.alpha, rasterized=True
                )
                ax.set_title(title)
                set_aspect_center(ax)
                self.add_lettering(ax, letter, fontdict=letter_dict, loc="left")
                if self.scalebars:
                    self.add_scalebar(ax, self.scalebars)
                else:
                    # Hide the right and top spines
                    ax.spines["right"].set_visible(False)
                    ax.spines["top"].set_visible(False)

                    ax.yaxis.set_ticks_position("left")
                    ax.xaxis.set_ticks_position("bottom")

        self.fig, self.axs = fig, axs
        return fig, axs

    @staticmethod
    def row_datapaths(
        datasource,
        k=None,
        perplexity=None,
        k_ratio=2,
        lo_exag=4,
        hi_exag=30,
        init=".",
        fa2_scale="stdscale;f:1e3",
    ):
        """Return a pair of 5-tuples that are five common algorithms
        on the same dataset.

        Returns
        -------
        A pair of data filenames and titles."""
        dsrc = Path(datasource)

        ann_prefix = "ann" + ("" if k is None else f";n_neighbors:{k}")
        umap_prefix = "umap_knn" + ("" if k is None else f";n_neighbors:{k}")

        if fa2_scale is None:
            fa2 = dsrc / ann_prefix / init / "fa2"
        elif isinstance(fa2_scale, (int, float)):
            fa2 = dsrc / ann_prefix / init / f"stdscale;f:{fa2_scale}" / "fa2"
        else:
            fa2 = dsrc / ann_prefix / init / fa2_scale / "fa2"
        umap = dsrc / umap_prefix / init / "maxscale;f:10" / "umap"

        tsne_prefix = "affinity"
        if perplexity is not None:
            tsne_prefix += f";perplexity:{perplexity}"
        elif k is not None:
            tsne_prefix += f";perplexity:{k_ratio*k}"

        tsne = dsrc / tsne_prefix / init / "stdscale;f:1e-4" / "tsne/"
        tsne4 = (
            dsrc
            / tsne_prefix
            / init
            / "stdscale;f:1e-4"
            / f"tsne;late_exaggeration:{lo_exag}"
        )
        tsne30 = (
            dsrc
            / tsne_prefix
            / init
            / "stdscale;f:1e-4"
            / f"tsne;early_exaggeration:{hi_exag};late_exaggeration:{hi_exag}"
        )
        tsnes = [tsne30, tsne4, tsne]

        titles = [
            "ForceAtlas2",
            "UMAP",
            # hack around to use the regular font for the numbers.
            # This uses dejavu font for \rho, which sucks, but is
            # hardly noticeable (fortunately).  The alternative would
            # be to use the upright version of the correct font by
            # specifying \mathdefault{\rho}
            f"$\\rho = {{}}${hi_exag}",
            f"$\\rho = {{}}${lo_exag}",
            "t-SNE",
        ]

        return [fa2, umap] + tsnes, titles


class SpectralVecs(ScatterMultiple):
    def __init__(
        self,
        paths,
        dataname=None,
        labelname=None,
        plotname=None,
        titles=None,
        lim_eps=0.025,
        alpha=0.5,
        format="png",
        rc=None,
        scalebars=None,
        **kwargs,
    ):
        super().__init__(
            paths,
            dataname=dataname,
            labelname=labelname,
            plotname=plotname,
            titles=titles,
            lim_eps=lim_eps,
            alpha=alpha,
            format=format,
            rc=rc,
            scalebars=scalebars,
            **kwargs,
        )

    def transform(self):
        spectral = self.data[0]

        if self.titles is None:
            titles = [f"Eigs {i+1}/{i+2}" for i in range(spectral.shape[1] - 1)]
        else:
            titles = self.titles

        rows, cols = auto_layout(spectral.shape[1] - 1)
        blocksize = self.figwidth / cols

        with plt.rc_context(fname=self.rc):
            fig, axs = plt.subplots(
                nrows=rows,
                ncols=cols,
                figsize=(blocksize * cols, blocksize * rows),
                squeeze=False,
                constrained_layout=True,
                **self.kwargs,
            )

            for i, (ax, title) in enumerate(zip(axs.flat, titles)):
                ax.scatter(
                    spectral[:, i],
                    spectral[:, i + 1],
                    c=self.labels,
                    alpha=self.alpha,
                    rasterized=True,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title)

            for ax in axs.flat[-(rows * cols - spectral.shape[1]) - 1 :]:
                ax.remove()

        self.fig, self.axs = fig, axs
        return fig, axs


class ExtPanelPlot(SixPanelPlot):
    """An extension of SixPanelPlot that will additionally add a
    measure of correlation in the lower-right axes."""

    def __init__(
        self,
        paths,
        corr_fname,
        dataname=None,
        labelname=None,
        plotname=None,
        titles=None,
        lim_eps=0.025,
        alpha=0.5,
        format="png",
        rc=None,
        scalebars=0.25,
        lo_exag=None,
        hi_exag=None,
        **kwargs,
    ):
        super().__init__(
            paths,
            dataname=dataname,
            labelname=labelname,
            plotname=plotname,
            titles=titles,
            lim_eps=lim_eps,
            alpha=alpha,
            format=format,
            rc=rc,
            scalebars=scalebars,
            **kwargs,
        )
        self.corr_fname = (
            corr_fname if corr_fname.is_absolute() else self.path / corr_fname
        )
        self.lo_exag = lo_exag
        self.hi_exag = hi_exag

    def get_datadeps(self):
        return [self.corr_fname] + super().get_datadeps()

    def load(self):
        self.labels, *self.data = [np.load(f) for f in self.get_datadeps()[2:]]
        self.df = pd.read_csv(self.corr_fname)
        df = self.df
        self.rhos = df["rho"]
        self.corrs = [df[["fa2", "umap"]].values]

    def transform(self):
        spectral, fa2, umap, tsne, tsne4, tsne30 = self.data
        labels = self.labels
        alpha = self.alpha

        width_inch = 5.5  # text (and thus fig) width for nips
        rows = 2
        cols = 4
        box_inch = width_inch / cols
        with plt.rc_context(fname=self.rc):
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(width_inch, rows * box_inch),
                dpi=600,
                constrained_layout=True,
            )
            t_sp, t_fa, t_umap, *t_tsne = self.titles

            axs[0, 0].set_label("spectral")
            axs[0, 0].set_title(t_sp)
            axs[0, 0].scatter(
                spectral[:, 0], spectral[:, 1], c=labels, alpha=alpha, rasterized=True
            )
            axs[0, 3].set_title(t_tsne[0])
            axs[0, 3].scatter(
                tsne[:, 0], tsne[:, 1], c=labels, alpha=alpha, rasterized=True
            )

            axs[1, 1].set_title(t_fa)
            axs[1, 1].scatter(
                fa2[:, 0], fa2[:, 1], c=labels, alpha=alpha, rasterized=True
            )
            axs[1, 2].set_title(t_umap)
            axs[1, 2].scatter(
                umap[:, 0], umap[:, 1], c=labels, alpha=alpha, rasterized=True
            )

            # add t-sne with exaggeration
            exag_axs = []
            for title, data, g in zip(
                t_tsne[1:], [tsne4, tsne30], [axs[0, 2], axs[0, 1]]
            ):
                ax = fig.add_subplot(g, zorder=5)
                exag_axs.append(ax)
                ax.set_title(title)
                ax.scatter(
                    data[:, 0], data[:, 1], c=labels, alpha=alpha, rasterized=True
                )

            axs[1, 0].set_label("legend")
            gs = axs[1, 3].get_gridspec()
            stat_gs = gs[1, 3].subgridspec(1, 1, wspace=0, hspace=0)
            ax = fig.add_subplot(stat_gs[0])
            ax.set_label("stats")
            axs[1, 3].set_label("stat_letter")
            plot_correlation_rhos(
                self.rhos, self.corrs, ax, lo_exag=self.lo_exag, hi_exag=self.hi_exag,
            )

            letter_iter = iter(string.ascii_lowercase)

            for ax in fig.get_axes():
                if (
                    self.scalebars
                    and ax.get_label() != "spectral"
                    and ax.get_label() != "legend"
                    and ax.get_label() != "stats"
                    and ax.get_label() != "stat_letter"
                ):
                    set_aspect_center(ax)
                    self.add_lettering(ax, next(letter_iter))
                    self.add_scalebar(ax, self.scalebars)
                elif ax.get_label() == "spectral":
                    set_aspect_center(ax)
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.set_frame_on(False)
                    self.add_lettering(ax, next(letter_iter))
                elif ax.get_label() == "legend":
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.set_frame_on(False)
                    # If we're plotting treutlein data, add the legend
                    # in the "legend" axes
                    if "treutlein" in [p[: len("treutlein")] for p in self.path.parts]:
                        add_treutlein_legend(ax, self.labels)
                    if "hydra" in [p[: len("hydra")] for p in self.path.parts]:
                        add_hydra_legend(ax, self.labels)
                    if "gauss_devel" in [
                        p[: len("gauss_devel")] for p in self.path.parts
                    ]:
                        add_gauss_devel_legend(ax, self.labels)
                elif ax.get_label() == "stat_letter":
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.set_frame_on(False)
                    self.add_lettering(ax, next(letter_iter))
                else:
                    # Hide the right and top spines
                    ax.spines["right"].set_visible(False)
                    ax.spines["top"].set_visible(False)

                    ax.yaxis.set_ticks_position("left")
                    ax.xaxis.set_ticks_position("bottom")

        self.fig, self.axs = fig, axs
        self.data_ = fig, axs
        return fig, axs


def titles_from_paths(paths):
    ns_dicts = [name_and_dict(p) for p in paths]
    name = ns_dicts[0][0]  # take the first algo name for comparing
    different_algos = not all([n == name for n, _ in ns_dicts])

    ret = []
    for name, kwargs in ns_dicts:
        s = name if different_algos else ""

        for k, v in kwargs.items():
            s += f" {k}={v}"
        s = s.strip()
        ret.append(s)

    return ret


def auto_layout(n):
    rows = cols = int(np.ceil(np.sqrt(n)))

    while rows * cols >= n:
        rows -= 1

    rows += 1

    # some layouts that look better in this configuration (imo)
    if n == 3:
        return (1, 3)
    elif n == 8:
        return (2, 4)
    else:
        return (rows, cols)


def set_aspect_center(ax):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.set_aspect(1)
    l = max(xmax - xmin, ymax - ymin)
    l2 = l / 2
    x = (xmax + xmin) / 2
    y = (ymax + ymin) / 2
    ax.axis([x - l2, x + l2, y - l2, y + l2])


def add_treutlein_legend(ax, labels):
    # 0 days, 4 days, 10 days, 15 days, 1 month, 2 months, 4 months
    lbl_map = {
        "  0 days": "navy",
        "  4 days": "royalblue",
        "10 days": "skyblue",
        "15 days": "lightgreen",
        "  1 month": "gold",
        "  2 months": "tomato",
        "  3 months": "firebrick",
        "  4 months": "maroon",
    }
    # lines.Line2D([], [], color="blue", marker="*", markersize=15, label="Blue stars")
    markers = [
        mpl.lines.Line2D(
            [],
            [],
            label=key,
            color=c,
            ls="",
            marker=mpl.rcParams["scatter.marker"],
            markersize=mpl.rcParams["font.size"],
        )
        for key, c in lbl_map.items()
        if c in labels
    ]
    legend = ax.legend(
        handles=markers, loc="center", fancybox=False, handletextpad=0.1,
    )
    legend.get_frame().set_linewidth(0.4)
    return legend


def add_hydra_legend(ax, labels):
    lbl_map = {
        "endoderm": "xkcd:light green",
        "ectoderm": "xkcd:sky blue",
        "interstitial": "xkcd:salmon",
    }

    markers = [
        mpl.lines.Line2D(
            [],
            [],
            label=key,
            color=c,
            ls="",
            marker=mpl.rcParams["scatter.marker"],
            markersize=mpl.rcParams["font.size"],
        )
        for key, c in lbl_map.items()
        if c in labels
    ]
    legend = ax.legend(
        handles=markers, loc="center", fancybox=False, handletextpad=0.1,
    )
    legend.get_frame().set_linewidth(0.4)
    return legend


def add_gauss_devel_legend(ax, labels, cmap="copper"):
    """Add a legend for the gauss_devel simulated data.  This has the
    hardcoded assumption that it's using the copper cmap."""
    rc = mpl.rcParams
    n = np.unique(labels, axis=0).shape[0]
    cm = plt.get_cmap(cmap, lut=n)

    steps = [1, "...", n // 2, "...", n]

    markers = [
        mpl.lines.Line2D(
            [],
            [],
            label=f"step {i}" if isinstance(i, int) else i,
            color=cm(i - 1) if isinstance(i, int) else None,
            ls="",
            marker=rc["scatter.marker"] if isinstance(i, int) else None,
            markersize=rc["font.size"],
        )
        for i in steps
    ]
    legend = ax.legend(
        handles=markers, loc="center", fancybox=False, handletextpad=0.1,
    )
    legend.get_frame().set_linewidth(0.4)
    return legend


def add_famnist_legend(ax):
    name_list = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sweater",
        "Bag",
        "Ankle Boots",
    ]
    rc = mpl.rcParams
    cm = plt.get_cmap(lut=len(name_list))
    markers = [
        mpl.lines.Line2D(
            [],
            [],
            label=label,
            color=cm(i),
            ls="",
            marker=rc["scatter.marker"],
            markersize=rc["font.size"],
        )
        for i, label in enumerate(name_list)
    ]
    legend = ax.legend(
        handles=markers, loc="center", fancybox=False, handletextpad=0.1,
    )
    legend.get_frame().set_linewidth(0.4)
    return legend


def despine(ax):
    """Remove the upper and right axes border."""
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
