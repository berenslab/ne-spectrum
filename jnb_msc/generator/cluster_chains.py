from .generator import GenStage

import numpy as np


def cluster_chains(
    n_clusters=10,
    dim=10,
    pts_cluster=100,
    pts_chain=50,
    dumbbell_offset=0.5,
    dumbbell_std=1 / 7.5,
    random_state=None,
):

    if random_state is None:
        rng = np.random.RandomState()
    else:
        rng = random_state

    n = n_clusters * pts_cluster + (n_clusters - 1) * pts_chain
    # X = np.empty((n, dim))
    y = np.zeros(n, dtype=np.uint)
    y[: n_clusters * pts_cluster] = 0

    means = np.eye(dim)
    cov = np.eye(dim) * dumbbell_std ** 2

    clusters = np.empty((n_clusters, pts_cluster, dim))
    chains = np.empty((n_clusters - 1, pts_chain, dim))
    for i in range(n_clusters):
        # multivariate_normal will put the dimension specified by mean and cov into axis -1
        dumbbell = rng.multivariate_normal(means[i], cov, size=(pts_cluster))
        dumbbell[: len(dumbbell) // 2, -1] -= dumbbell_offset / 2
        dumbbell[len(dumbbell) // 2 :, -1] += dumbbell_offset / 2
        clusters[i] = dumbbell
        if i + 1 < n_clusters:
            chains[i] = np.linspace(
                means[i], means[i + 1], num=pts_chain, endpoint=True
            )  # + random_state.normal(0, 0.01, size=(pts_chain, dim))

        y[i * pts_cluster : (i + 1) * pts_cluster] = i + 1

    X = np.vstack([clusters.reshape(-1, dim), chains.reshape(-1, dim)])
    return X, y


class ClusterChains(GenStage):
    def __init__(
        self,
        path,
        dataname="data.npy",
        labelname="labels.npy",
        descname="descr.md",
        random_state=None,
        n_clusters=10,
        dim=50,
        pts_cluster=500,
        pts_chain=300,
        dumbbell_dist=1.25,  # 1.25 for 50d
    ):
        super().__init__(
            path, dataname=dataname, labelname=labelname, random_state=random_state
        )
        self.n_clusters = n_clusters
        if dim is None:
            self.dim = self.n_clusters + 1
        else:
            self.dim = dim
        self.pts_cluster = pts_cluster
        self.pts_chain = pts_chain
        self.dumbbell_dist = dumbbell_dist

    def transform(self):
        self.data_, self.labels_ = cluster_chains(
            self.n_clusters,
            self.dim,
            self.pts_cluster,
            self.pts_chain,
            dumbbell_offset=self.dumbbell_dist,
            random_state=self.random_state,
        )

        self.description_ = (
            "Clusters on a chain. The clusters are Gaussian blobs over all "
            "dimensions and they're connected via chains.  In total it "
            f"consists of {self.data_.shape[0]} points.  Each cluster has "
            f"{self.pts_cluster} points and each chain consist of "
            f"{self.pts_chain} points."
        )

        return self.data_
