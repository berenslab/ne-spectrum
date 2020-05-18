from ..abpc import ProjectBase

import numpy as np

from scipy.io import mmread, mmwrite
from pathlib import Path


class TransformerStage(ProjectBase):
    def __init__(
        self,
        path,
        dataname="data.npy",
        initname=None,
        outname=None,
        descname=None,
        n_components=2,
        random_state=None,
    ):
        super().__init__(path=path, random_state=random_state)
        self.dataname = dataname
        self.initname = dataname if initname is None else initname
        self.outname = dataname if outname is None else outname
        self.descname = "descr.md" if descname is None else descname
        self.n_components = n_components

    def get_datadeps(self):
        return [self.indir / self.dataname, self.indir / self.descname]

    def load(self):
        self.data = np.load(self.get_datadeps()[0])
        # unused as of now
        self.description = Path(self.get_datadeps()[1]).read_text()


class NDStage(TransformerStage):
    """Perform dimensionality reduction on an NxD matrix."""

    def save(self):
        self.save_lambda(self.outdir / self.outname, self.data_, np.save)


class NNStage(TransformerStage):
    """Transform an NxD matrix into an NxN matrix."""

    def __init__(self, path, dataname="data.npy", outname="nns.mtx", random_state=None):
        # override outname
        super().__init__(
            path=path, dataname=dataname, outname=outname, random_state=random_state
        )

    def save(self):
        self.save_lambda(self.outdir / self.outname, self.data_, mmwrite)


class SimStage(TransformerStage):
    """Transform an initial layout via an NN matrix to another
    embedding."""

    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        descname=None,
        n_components=2,
        random_state=None,
    ):
        # override initname
        super().__init__(
            path=path,
            dataname=dataname,
            initname=initname,
            outname=None,
            descname=descname,
            n_components=n_components,
            random_state=random_state,
        )

    def get_datadeps(self):
        return [self.indir / self.dataname, self.indir / self.initname]

    def load(self):
        inlist = self.get_datadeps()
        self.data = mmread(str(inlist[0]))
        self.init = np.load(inlist[1])[:, : self.n_components]

    def save(self):
        self.save_lambda(self.outdir / self.initname, self.data_, np.save)
