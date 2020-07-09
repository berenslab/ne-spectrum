from ..plot import ScatterSingle, ScatterMultiple
from ..plot import auto_layout, titles_from_paths

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mplclrs
import matplotlib.animation as animation
import sys
from pathlib import Path


class ScatterAnimation(ScatterSingle):
    """Transform a sequence of npy-arrays (specified in an .flist
    file) into an animation."""

    def __init__(
        self,
        path,
        dataname="data.flist",
        labelname=None,
        plotname="data.mp4",
        lim_eps=0.025,
        alpha=0.5,
        format="png",
        rc=None,
        fps=50,
        **kwargs,
    ):
        super().__init__(
            path,
            dataname=dataname,
            labelname=labelname,
            plotname=plotname,
            lim_eps=lim_eps,
            alpha=alpha,
            format=format,
            rc=rc,
            **kwargs,
        )
        self.fps = fps
        # if the wrong rc file has been loaded, try to rectify that
        if isinstance(self.rc, Path) and not self.rc.is_file():
            if (self.rc.parent.parent / "matplotlibrc").is_file():
                self.rc = self.rc.parent.parent / "matplotlibrc"
            else:
                self.rc = None

    def get_datadeps(self):
        return [self.rc, self.labelname, self.dataname]

    def load(self):
        self.labels = np.load(self.labelname)
        with open(self.dataname) as f:
            self.datafiles = [Path(l.strip()) for l in f]

        self.data = np.load(self.datafiles[0])

    def transform(self):

        with plt.rc_context(fname=self.rc):
            fig, ax = plt.subplots(
                figsize=(2, 2), dpi=200, constrained_layout=True, **self.kwargs
            )

            d = self.data
            self.data = (d - d.min()) / (d.max() - d.min())
            sc = ax.scatter(
                self.data[:, 0], self.data[:, 1], c=self.labels, alpha=self.alpha
            )

            ax.set_xlim(*self.lim(0, 1, self.lim_eps))
            ax.set_ylim(*self.lim(0, 1, self.lim_eps))

            ax.set_axis_off()

            sc_ani = animation.FuncAnimation(
                fig,
                update_scatter,
                frames=load_files(self.datafiles),
                fargs=([sc],),
                interval=self.fps,
                blit=True,
                init_func=lambda: [sc],
                save_count=len(self.datafiles),
            )

            # save in transform because this is the computationally
            # expensive operation
            sc_ani.save(
                str(self.plotname),
                fps=self.fps,
                metadata={"artist": "Niklas Böhm"},
                savefig_kwargs={"transparent": True},
            )

    def save(self):
        pass


class ScatterAnimations(ScatterMultiple):
    def __init__(
        self,
        paths,
        dataname="data.flist",
        labelname=None,
        plotname="data.mp4",
        lim_eps=0.025,
        alpha=0.5,
        format="mp4",
        rc=None,
        **kwargs
        fps=50,
        dpi=100,
        **kwargs,
    ):
        super().__init__(
            paths,
            dataname=dataname,
            labelname=labelname,
            plotname=plotname,
            lim_eps=lim_eps,
            alpha=alpha,
            format=format,
            rc=rc,
            **kwargs
        )
        self.fps = fps
        self.dpi = dpi
        # if the wrong rc file has been loaded, try to rectify that
        if isinstance(self.rc, Path) and not self.rc.is_file():
            if (self.rc.parent.parent / "matplotlibrc").is_file():
                self.rc = self.rc.parent.parent / "matplotlibrc"
            else:
                self.rc = None

    def load(self):
        self.labels = np.load(self.labelname)
        self.dataffiles = []
        self.inits = []
        for flist in self.get_datadeps()[2:]:
            with open(flist) as f:
                self.dataffiles.append([Path(l.strip()) for l in f])
            self.inits.append(np.load(self.dataffiles[-1][0]))

    def transform(self):

        with plt.rc_context(fname=self.rc):
            rows, cols = auto_layout(len(self.dataffiles))
            titles = titles_from_paths(self.paths)
            fig, axs = plt.subplots(
                nrows=rows,
                ncols=cols,
                figsize=(2 * cols, 2 * rows),
                constrained_layout=True,
                dpi=self.dpi,
                **self.kwargs,
            )

            scatters = []
            for ax, data, title in zip(axs.flat, self.inits, titles):
                d = rescale(data)
                sc = ax.scatter(d[:, 0], d[:, 1], c=self.labels, alpha=self.alpha)
                scatters.append(sc)

                ax.set_xlim(self.lim(0, 1, self.lim_eps))
                ax.set_ylim(self.lim(0, 1, self.lim_eps))

                ax.set_axis_off()

                if title != "":
                    ax.set_title(title)

            fig.set_constrained_layout(False)
            sc_ani = animation.FuncAnimation(
                fig,
                update_scatters,
                frames=loader(self.dataffiles),
                fargs=(scatters,),
                interval=self.fps,
                blit=True,
                save_count=len(self.dataffiles[0]),
            )

            # save in transform because this is the computationally
            # expensive operation
            save = lambda f, anim: anim.save(
                f.name,  # f is a NamedTemporaryFile
                fps=self.fps,
                metadata={"artist": "Niklas Böhm"},
                extra_args=["-f", self.format],
            )
            self.save_lambda(self.plotname, sc_ani, save)

    def save(self):
        pass


def update_scatter(data, scatter):
    scatter[0].set_offsets(data)
    return scatter


def update_scatters(data, scatters):
    [s.set_offsets(d) for d, s in zip(data, scatters)]
    return scatters


def loader(ffiles):
    ars = []
    n = len(ffiles[0])
    for i in range(n):
        yield [rescale(np.load(flist[i])) for flist in ffiles]


def rescale(ar):
    """Rescale the given numpy array to fit into the range [0,1]."""
    return (ar - ar.min()) / (ar.max() - ar.min())


def load_files(files):
    for f in files:
        yield rescale(np.load(f)[:, :2])
