from .transformer import SimStage

from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.manifold._spectral_embedding import _set_diag
from sklearn.utils.validation import check_array
from pyamg import smoothed_aggregation_solver
from sklearn.externals._lobpcg import lobpcg
from scipy.io import mmread, mmwrite
from scipy.sparse.linalg import eigsh

import numpy as np
import scipy.sparse as sparse


class Spectral(SimStage):
    """Transforms an NxN matrix into a layout.

    In contrast to the other subclasses of `SimStage`, the Spectral
    embedding does not rely on an initial layout for the computation."""

    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=15,
        random_state=None,
        norm_laplacian=True,
        drop_first=True,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
        )
        self.norm_laplacian = norm_laplacian
        self.drop_first = drop_first

    def load(self):
        self.data = mmread(str(self.indir / self.dataname))

    def transform(self):
        # self.n_components = min(self.n_components, self.data.shape[1])
        laplacian, dd = csgraph_laplacian(
            self.data, normed=self.norm_laplacian, return_diag=True
        )
        laplacian = check_array(laplacian, dtype=np.float64, accept_sparse=True)
        laplacian = _set_diag(laplacian, 1, self.norm_laplacian)

        ## Seed the global number generator because the pyamg
        ## interface apparently uses that...
        ## Also, see https://github.com/pyamg/pyamg/issues/139
        np.random.seed(self.random_state.randint(2 ** 31 - 1))

        diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
        laplacian += diag_shift
        ml = smoothed_aggregation_solver(check_array(laplacian, "csr"))
        laplacian -= diag_shift

        M = ml.aspreconditioner()
        X = self.random_state.rand(laplacian.shape[0], self.n_components + 1)
        X[:, 0] = dd.ravel()

        # laplacian *= -1
        # v0 = self.random_state.uniform(-1, 1, laplacian.shape[0])
        # eigvals, diffusion_map = eigsh(
        #     laplacian, k=self.n_components + 1, sigma=1.0, which="LM", tol=0.0, v0=v0
        # )
        # # eigsh needs reversing
        # embedding = diffusion_map.T[::-1]

        eigvals, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.0e-5, largest=False)
        embedding = diffusion_map.T
        if self.norm_laplacian:
            embedding = embedding / dd

        if self.drop_first:
            self.data_ = embedding[1 : self.n_components].T
            eigvals = eigvals[1 : self.n_components]
        else:
            self.data_ = embedding[: self.n_components].T

        self.eigvals_ = eigvals[::-1]  # reverse direction to have the largest first

    def save(self):
        super().save()
        np.save(self.outdir / "eigvals.npy", self.eigvals_)
