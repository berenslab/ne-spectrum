from .transformer import NNStage

from scipy.sparse import lil_matrix, csr_matrix

import openTSNE
import annoy
import numpy as np


def make_adj_mat(
    X,
    n_neighbors=15,
    metric="euclidean",
    n_trees=50,
    seed=None,
    use_dists=False,
    symmetrize=True,
    drop_first=True,
):
    t = annoy.AnnoyIndex(X.shape[1], metric)
    if seed is not None:
        t.set_seed(seed)

    [t.add_item(i, x) for i, x in enumerate(X)]
    t.build(n_trees)

    # construct the adjacency matrix for the graph
    adj = lil_matrix((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        neighs_, dists_ = t.get_nns_by_item(i, n_neighbors + 1, include_distances=True)
        if drop_first:
            neighs = neighs_[1:]
            dists = dists_[1:]
        else:
            neighs = neighs_[:n_neighbors]
            dists = dists_[:n_neighbors]

        adj[i, neighs] = dists if use_dists else 1
        if symmetrize:
            adj[neighs, i] = dists if use_dists else 1  # symmetrize on the fly

    return adj, t


class ANN(NNStage):
    def __init__(
        self,
        path,
        dataname="data.npy",
        outname="nns.mtx",
        indexname="index.ann",
        knn_indices_name="knn_indices.npy",
        knn_dists_name="knn_dists.npy",
        random_state=None,
        n_neighbors=15,
        metric="euclidean",
        n_trees=50,
        use_dists=False,
    ):
        super().__init__(
            path=path, dataname=dataname, outname=outname, random_state=random_state
        )
        self.indexname = indexname
        self.knn_indices_name = knn_indices_name
        self.knn_dists_name = knn_dists_name

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.use_dists = use_dists

    def transform(self):
        seed = self.random_state.randint(-(2 ** 31), 2 ** 31)
        self.data_, self.annoy_idx = make_adj_mat(
            self.data,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_trees=self.n_trees,
            seed=seed,
            use_dists=self.use_dists,
        )

    def save(self):
        super().save()
        self.annoy_idx.save(str(self.outdir / self.indexname))


class AsymmetricANN(ANN):
    def transform(self):
        seed = self.random_state.randint(-(2 ** 31), 2 ** 31)
        self.data_, self.annoy_idx = make_adj_mat(
            self.data,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_trees=self.n_trees,
            seed=seed,
            use_dists=self.use_dists,
            symmetrize=False,
        )


class KNNAffinities(ANN):
    def __init__(
        self,
        path,
        dataname="data.npy",
        outname="nns.mtx",
        indexname="index.ann",
        knn_indices_name="knn_indices.npy",
        knn_dists_name="knn_dists.npy",
        random_state=None,
        n_neighbors=15,
        metric="euclidean",
        metric_arams=None,
        n_trees=50,
        n_jobs=1,
    ):
        super().__init__(
            path,
            dataname=dataname,
            outname=outname,
            indexname=indexname,
            knn_indices_name=knn_indices_name,
            knn_dists_name=knn_dists_name,
            random_state=random_state,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
        )
        self.n_jobs = n_jobs
        self.metric_params = None

    def transform(self):
        self.knn_idx, neighbors, distances = openTSNE.affinity.build_knn_index(
            self.data,
            "annoy",
            self.n_neighbors,
            self.metric,
            self.metric_params,
            self.n_jobs,
            self.random_state,
            False,
        )
        self.annoy_idx = self.knn_idx.index

        n = self.data.shape[0]
        A = csr_matrix(
            (
                np.ones_like(distances).ravel(),
                neighbors.ravel(),
                range(0, n * self.n_neighbors + 1, self.n_neighbors),
            ),
            shape=(n, n),
        )

        # Symmetrize + normalize the probability matrix
        P = (A + A.T) / 2
        P /= np.sum(P)
        self.data_ = P
