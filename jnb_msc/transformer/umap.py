from .transformer import SimStage, NDStage, NNStage
from .tsne import TSNESaveEmbedding

from scipy.io import mmread

import numba
import warnings
import umap.umap_ as umaplib
import umap.layouts
import numpy as np
import scipy.sparse as sp


def _optimize_layout_euclidean_single_epoch(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma,
    dim,
    move_other,
    alpha,
    eps,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = head_embedding[j]
            other = tail_embedding[k]

            dist_squared = umap.layouts.rdist(current, other)

            if dist_squared > 0.0:
                grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
                grad_coeff /= a * pow(dist_squared, b) + 1.0
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = umap.layouts.clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                if move_other:
                    other[d] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = umap.layouts.tau_rand_int(rng_state) % n_vertices

                other = tail_embedding[k]

                dist_squared = umap.layouts.rdist(current, other)

                if dist_squared > 0.0:
                    grad_coeff = 2.0 * gamma * b
                    grad_coeff /= (eps + dist_squared) * (a * pow(dist_squared, b) + 1)
                elif j == k:
                    continue
                else:
                    grad_coeff = 0.0

                for d in range(dim):
                    if grad_coeff > 0.0:
                        grad_d = umap.layouts.clip(grad_coeff * (current[d] - other[d]))
                    else:
                        grad_d = 4.0
                    current[d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


def optimize_layout_euclidean(
    head_embedding,
    tail_embedding,
    head,
    tail,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    gamma=1.0,
    initial_alpha=1.0,
    eps=0.001,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    saver=None,
):
    """This is a slightly edited copy from umap.layouts, amended for
    the use case here.  Look at the documentation in the respective
    function.

    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized embedding.
    """

    dim = head_embedding.shape[1]
    move_other = head_embedding.shape[0] == tail_embedding.shape[0]
    alpha = initial_alpha

    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
        _optimize_layout_euclidean_single_epoch, fastmath=True, parallel=parallel,
    )
    for n in range(n_epochs):
        optimize_fn(
            head_embedding,
            tail_embedding,
            head,
            tail,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            dim,
            move_other,
            alpha,
            eps,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            n,
        )

        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

        if callable(saver):
            saver(n, np.nan, head_embedding)

        if verbose and n % int(n_epochs / 10) == 0:
            print("\tcompleted ", n, " / ", n_epochs, "epochs")

    return head_embedding


class UMAP(SimStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        knn_indices_name="knn_indices.npy",
        knn_dists_name="knn_dists.npy",
        n_components=2,
        random_state=None,
        n_iter=500,
        learning_rate=1,
        a=1,
        b=1,
        nu=5,
        gamma=1,
        eps=0.001,
        parallel=False,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
        )
        self.knn_indices_name = knn_indices_name
        self.knn_dists_name = knn_dists_name

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.a = a
        self.b = b
        self.nu = nu
        self.parallel = parallel
        self.gamma = gamma
        self.eps = eps

    def get_datadeps(self):
        return [
            self.indir / self.dataname,
            self.indir / self.initname,
        ]

    def load(self):
        inlist = self.get_datadeps()

        self.data = mmread(str(inlist[0]))
        self.init = np.load(inlist[1])[:, : self.n_components]
        # This is the format that is needed for the numba functions
        self.init = np.array(self.init, copy=False, dtype=np.float32, order="C")

    def transform(self):
        self.graph_ = self.data

        epochs_per_sample = umaplib.make_epochs_per_sample(
            self.graph_.data, self.n_iter
        )

        self.graph_ = self.graph_.tocoo()
        n_vertices = self.graph_.shape[1]
        head = self.graph_.row
        tail = self.graph_.col

        saver = TSNESaveEmbedding(self.outdir)
        rng_state = self.random_state.randint(-(2 ** 31) + 1, 2 ** 31 - 1, 3).astype(
            np.int64
        )

        self.data_ = optimize_layout_euclidean(
            self.init,
            self.init,
            head,
            tail,
            self.n_iter,
            n_vertices,
            epochs_per_sample,
            self.a,
            self.b,
            rng_state,
            gamma=float(self.gamma),
            initial_alpha=self.learning_rate,
            eps=self.eps,
            negative_sample_rate=self.nu,
            parallel=self.parallel,
            verbose=False,
            saver=saver,
        )
        return self.data_


class UMAPKNN(NNStage):
    def __init__(
        self,
        path,
        dataname="data.npy",
        outname="nns.mtx",
        random_state=None,
        n_neighbors=15,
        metric="euclidean",
        angular=False,
        set_op_mix_ratio=1,
        local_connectivity=1,
    ):
        super().__init__(
            path, dataname=dataname, outname=outname, random_state=random_state
        )
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.angular = angular
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity

    def transform(self):

        self.data_, self._sigmas, self._rhos = umaplib.fuzzy_simplicial_set(
            X=self.data,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            metric=self.metric,
            angular=self.angular,
            set_op_mix_ratio=self.set_op_mix_ratio,
            local_connectivity=self.local_connectivity,
        )


class UMAPDefault(NDStage):
    def __init__(
        self,
        path,
        dataname="data.npy",
        initname=None,
        outname=None,
        descname=None,
        n_components=2,
        random_state=None,
        n_iter=500,
        learning_rate=1,
        a=1,
        b=1,
        k=15,
        nu=5,
        gamma=1,
        metric="euclidean",
        angular=False,
        set_op_mix_ratio=1,
        local_connectivity=1,
        parallel=True,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
        )

        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.a = a
        self.b = b
        self.k = k
        self.nu = nu
        self.metric = metric
        self.angular = angular
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.parallel = parallel
        self.gamma = gamma

        if self.metric != "euclidean":
            raise RuntimeError(
                "This module makes the assumption that euclidean metric "
                " is used, but {} was passed".format(self.metric)
            )

    def get_datadeps(self):
        return [
            self.indir / self.initname,
            self.indir / self.descname,
        ]

    def load(self):
        self.data = np.load(self.indir.parent.parent / self.dataname)
        self.init = np.load(self.get_datadeps()[0])[:, : self.n_components]

    def transform(self):
        vis = umaplib.UMAP(
            self.k,
            self.n_components,
            self.metric,
            n_epochs=self.n_iter,
            learning_rate=self.learning_rate,
            init=self.init,
            a=self.a,
            b=self.b,
            repulsion_strength=self.gamma,
            negative_sample_rate=self.nu,
            local_connectivity=self.local_connectivity,
            set_op_mix_ratio=self.set_op_mix_ratio,
            random_state=self.random_state,
        )

        self.data_ = vis.fit_transform(self.data)
        return self.data_
