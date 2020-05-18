from . import SimStage
from .tsne import TSNESaveEmbedding

import numpy as np

import numba

from pathlib import Path


class AttractionRepulsionModel(SimStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        knn_indices_name="knn_indices.npy",
        knn_dists_name="knn_dists.npy",
        a=1,
        r=-1,  # FA2 default
        n_iter=250,
        repulsion=1,
        learning_rate=0.1,
        momentum=0.8,
        anneal="linear",
        max_grad_norm=1000,
    ):
        super().__init__(path, dataname, initname, n_components, random_state)
        self.knn_indices_name = knn_indices_name
        self.knn_dists_name = knn_dists_name

        self.a = a
        self.r = r
        self.n_iter = n_iter
        self.repulsion = repulsion
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.anneal = anneal
        self.max_grad_norm = max_grad_norm

        self.saver = TSNESaveEmbedding(self.outdir)
        self.gradsaver = TSNESaveEmbedding(self.outdir / "grad")
        self.lrs = []

        if self.anneal == "linear":
            self.lr_adjust = lambda lr, i, n: lr * (1 - i / n)
        elif self.anneal == "linhalf":
            self.lr_adjust = lambda lr, i, n: lr * (1 - (i / n) / 2)
        else:
            self.lr_adjust = lambda lr, i, n: lr

    def get_datadeps(self):
        return [
            self.indir / self.knn_indices_name,
            self.indir / self.knn_dists_name,
            self.indir / self.initname,
        ]

    def load(self):
        inlist = self.get_datadeps()

        self.knn_indices = np.load(inlist[0])
        self.knn_dists = np.load(inlist[1])
        self.knn_dists[self.knn_dists != 0] = 1
        self.data = self.knn_indices, self.knn_dists
        self.init = np.load(inlist[2])[:, : self.n_components]
        # This is the format that is needed for the numba functions
        self.init = np.array(self.init, copy=False, dtype=np.float32, order="C")

    def transform(self):
        self.data_ = self.init

        import sys

        grads = np.zeros((2, *self.data_.shape))
        self.saver(-1, np.nan, self.data_)  # save the initial layout
        for n in range(self.n_iter):
            grad = grads[n % 2]
            grad_old = grads[(n + 1) % 2]
            grad = energy_gradient(
                grad,
                self.knn_indices,
                self.knn_dists,
                self.data_,
                self.a,
                self.r,
                self.max_grad_norm,
                self.repulsion,
            )
            # print(grad[n % grad.shape[0]], file=sys.stderr)
            print((grad ** 2).mean(), file=sys.stderr)
            self.saver(n, np.nan, self.data_)
            self.gradsaver(n, np.nan, grad)

            self.data_ += self.learning_rate * (grad + self.momentum * grad_old)

            self.lrs.append(self.learning_rate)
            self.learning_rate = self.lr_adjust(self.learning_rate, n, self.n_iter)
        return self.data_

    def save(self):
        super().save()
        self.save_lambda(self.outdir / "lrs.npy", self.lrs, np.save)


@numba.njit(parallel=True)
def energy_gradient(
    grad, knn_indices, knn_dists, layout, a, r, max_grad_norm=1000, repulsion_factor=1
):
    grad[:] = 0
    for i in numba.prange(layout.shape[0]):
        for j in range(layout.shape[0]):
            if i == j:
                continue

            diff = layout[j] - layout[i]
            diffnorm = np.linalg.norm(diff)

            if diffnorm <= 1e-120:
                continue

            # attraction
            if (knn_indices[i] == j).any() or (knn_indices[j] == i).any():
                grad[i] += diff / diffnorm * 1 * diffnorm ** a

            # repulsion
            # knn_dists[i].sum()  # degree of node i
            grad[i] -= (
                knn_dists[i].sum()
                * knn_dists[j].sum()
                * diff
                / diffnorm
                * diffnorm ** r
                * repulsion_factor
            )

            # clip gradients
            grad_i = grad[i]
            grad_i[grad[i] > max_grad_norm] = max_grad_norm
            grad_i[grad[i] < -max_grad_norm] = -max_grad_norm
    return grad
