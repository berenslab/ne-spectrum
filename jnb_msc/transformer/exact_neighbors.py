from .transformer import NNStage

from scipy.sparse import lil_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors

import numpy as np


def make_adj_mat(
    X,
    n_neighbors=15,
    metric="euclidean",
    use_dists=False,
    symmetrize=True,
    drop_first=True,
):
    nns = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nns.fit(X)

    # construct the adjacency matrix for the graph
    adj = lil_matrix((X.shape[0], X.shape[0]))
    dists_, neighs_ = nns.kneighbors(X=None, n_neighbors=n_neighbors, return_distance=True)
    idx_self = np.repeat(np.arange(0, X.shape[0]), n_neighbors)
    adj[idx_self, neighs_.flat] = dists_ if use_dists else 1
    if symmetrize:
        adj = (adj + adj.T) / 2

    return adj


class ExactNeighbors(NNStage):
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
        self.use_dists = use_dists

    def transform(self):
        self.data_ = make_adj_mat(
            self.data,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            use_dists=self.use_dists,
        )
