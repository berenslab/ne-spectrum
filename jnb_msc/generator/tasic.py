import numpy as np

from . import GenStage


class TasicMouse(GenStage):
    """Data of the developmental data by Treutlein et al."""

    def load(self):
        self.data = np.load(self.path / "../../static/tasic.data.npy")
        self.labels = np.load(self.path / "../../static/tasic.labels.npy")
        self.description_ = "Dataset of an adult mouse cortex from Tasic et al. 2018."

    def transform(self):
        self.labels_ = self.labels
        self.data_ = self.data

        return self.data_
