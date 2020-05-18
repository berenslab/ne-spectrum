from .generator import GenStage

import numpy as np
import matplotlib.pyplot as plt


def gauss_clusters(
    n_clusters=10, dim=10, pts_cluster=100, random_state=None, cov=1, stepsize=1,
):

    if random_state is None:
        rng = np.random.RandomState()
    else:
        rng = random_state

    n = n_clusters * pts_cluster

    s = stepsize / np.sqrt(dim)
    means = np.linspace(np.zeros(dim), n_clusters * s, num=n_clusters, endpoint=False)
    cshift_mask = np.zeros(n_clusters, dtype=np.bool)
    cshift_mask[15] = True
    cov = np.eye(dim) * cov

    clusters = np.array(
        [rng.multivariate_normal(m, cov, size=(pts_cluster)) for m in means]
    )

    X = np.reshape(clusters, (-1, dim))

    y = np.repeat(np.arange(n_clusters), pts_cluster)
    return X, y


class GaussLine(GenStage):
    def __init__(
        self,
        path,
        dataname="data.npy",
        labelname="labels.npy",
        descname="descr.md",
        random_state=None,
        n_clusters=20,
        dim=50,
        pts_cluster=1000,
        cluster_dist=6,
        cmap="copper",
    ):
        super().__init__(
            path, dataname=dataname, labelname=labelname, random_state=random_state
        )
        self.n_clusters = n_clusters
        self.dim = dim
        self.pts_cluster = pts_cluster
        self.cluster_dist = cluster_dist
        self.cmap = cmap

    def transform(self):
        self.data_, self.labels = gauss_clusters(
            self.n_clusters,
            self.dim,
            self.pts_cluster,
            stepsize=self.cluster_dist,
            random_state=self.random_state,
        )

        # transform the labels to proper rgb colors already so we
        # don't have to worry about it later
        self.cmap = plt.get_cmap(self.cmap, lut=self.labels.max() + 1)
        self.labels_ = self.cmap(self.labels)

        self.description_ = (
            "Gaussian clusters shifted along a line.  This dataset is "
            "simulating developmental data."
        )

        return self.data_
