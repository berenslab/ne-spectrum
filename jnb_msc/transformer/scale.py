from .transformer import NDStage

import numpy as np

from sklearn.preprocessing import normalize
from scipy.io import mmread, mmwrite


class AbstractScale(NDStage):
    def __init__(
        self,
        path,
        dataname="data.npy",
        initname=None,
        outname=None,
        random_state=None,
        f=1,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            outname=outname,
            random_state=random_state,
        )
        self.f = f


class StdScale(AbstractScale):
    """Scales its input to have an std equal to the given factor `f`."""

    def transform(self):
        self.data -= self.data.mean()
        self.data_ = self.data * (self.f / np.std(self.data, axis=0).max())


class MaxScale(AbstractScale):
    """Scale input to be in the range `[-f, f]`, for a given `f`.

    If only nonnegative values are allowed, pass
    `negative_rage=False`.  This will scale the input to be in `[0, f]`"""

    def __init__(
        self,
        path,
        dataname="data.npy",
        initname=None,
        outname=None,
        random_state=None,
        f=1,
        negative_range=False,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            outname=outname,
            random_state=random_state,
            f=f,
        )
        self.negative_range = negative_range

    def transform(self):
        if not self.negative_range:
            self.data -= self.data.min(axis=0)
        self.data_ = (self.data / np.abs(self.data).max(axis=0)) * self.f

        return self.data_


class SparseScale(AbstractScale):
    """Operate on a sparse matrix."""

    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname=None,
        outname="nns.mtx",
        random_state=None,
        f=1,
        sparse=True,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            outname=outname,
            random_state=random_state,
            f=f,
        )
        self.sparse = sparse

    def load(self):
        if self.sparse:
            self.data = mmread(str(self.indir / self.dataname))
        else:
            self.data = np.load(self.indir / self.dataname)

    def save(self):
        self.save_lambda(
            self.outdir / self.outname, self.data_, mmwrite if self.sparse else np.save
        )


class AxisNormalize(SparseScale):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname=None,
        outname="nns.mtx",
        random_state=None,
        f=1,
        sparse=True,
        axis=None,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            outname=outname,
            random_state=random_state,
            f=f,
            sparse=sparse,
        )
        self.axis = axis

    def transform(self):
        if self.axis is not None:
            self.data_ = normalize(self.data, norm="l1", axis=self.axis)
        else:
            self.data_ = self.data / self.data.sum()
        return self.data_


class RowNormalize(AxisNormalize):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname=None,
        outname="nns.mtx",
        random_state=None,
        f=1,
        sparse=True,
        axis=1,  # override axis
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            outname=outname,
            random_state=random_state,
            f=f,
            sparse=sparse,
            axis=axis,
        )


class ScalarScale(SparseScale):
    def transform(self):
        self.data_ = self.data * self.f
        return self.data_


class InvScale(SparseScale):
    def transform(self):
        self.data_ = self.data
        # access the dense part of the sparse mat and do element-wise inversion
        self.data_.data = self.data.data ** -1
        return self.data_


class Symmetrize(SparseScale):
    """Symmetrize a kNN matrix in the t-SNE fashion: (A + A^t) / 2."""

    def transform(self):
        self.data_ = (self.data + self.data.T) / 2
        return self.data_
