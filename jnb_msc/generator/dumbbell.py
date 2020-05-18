import numpy as np

from . import GenStage

from scipy.io import mmwrite


class Dumbbell(GenStage):
    """A dataset generator for a tiny dataset.

    Hardcoded values as provided by Dmitry."""

    def load(self):
        self.adj = np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 0],
            ]
        )

        self.knn_ind = np.array(
            [[1, 2, -1], [0, 2, -1], [0, 1, 2], [2, 4, 5], [3, 5, -1], [3, 4, -1]]
        )
        self.knn_dists = np.ones_like(self.knn_ind)

    def transform(self):
        self.data_ = self.random_state.randn(self.adj.shape[0], 2)
        self.labels_ = np.arange(self.adj.shape[0])
        self.description_ = (
            f"A dumbbell consisting of {self.data_.shape[0]} points, "
            "which are triangles connected with an edge."
        )

        return self.data_

    def save(self):
        super().save()
        self.save_lambda(self.outdir / "nns.mtx", self.adj, mmwrite)
        self.save_lambda(self.outdir / "knn_indices.npy", self.knn_ind, np.save)
        self.save_lambda(self.outdir / "knn_dists.npy", self.knn_dists, np.save)
