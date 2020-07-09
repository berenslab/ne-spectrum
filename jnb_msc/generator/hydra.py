import numpy as np

from . import GenStage


class HydraTrancriptomic(GenStage):
    """Data of the developmental data by Siebert et al.

    https://science.sciencemag.org/content/365/6451/eaav9314.long"""

    def load(self):
        self.data = np.load(self.path / "../../static/hydra.data.npy")
        self.labels = np.load(self.path / "../../static/hydra.labels.npy")
        self.description_ = (
            "Dataset of a hydra differentiation trajectory from Siebert et al. 2019."
        )

    def transform(self):
        self.labels_ = self.labels
        self.data_ = self.data

        return self.data_
