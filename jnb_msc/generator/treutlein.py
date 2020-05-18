import numpy as np

from . import GenStage


class TreutleinChimp(GenStage):
    """Data of the developmental data by Treutlein et al."""

    def load(self):
        self.data = np.load(self.path / '../../static/chimp.data.npy')
        self.labels = np.load(self.path / '../../static/chimp.labels.npy')
        self.description_ = (
            "Dataset of developmental data of a chimpanzee.  "
            f"Preprocessed and reduced to {self.data.shape[1]} "
            "dimension."
        )

    def transform(self):
        self.lbl_map = {
            "iPSCs": "navy",
            "EB": "royalblue",
            "Neuroectoderm": "skyblue",
            "Neuroepithelium": "lightgreen",
            "Organoid-1M": "gold",
            "Organoid-2M": "tomato",
            "Organoid-3M": "firebrick",
            "Organoid-4M": "maroon",
        }

        self.labels_ = [self.lbl_map[l] for l in self.labels]
        self.data_ = self.data

        return self.data_


class TreutleinHumanH9(TreutleinChimp):
    def load(self):
        self.data = np.load(self.path / '../../static/human-h9.data.npy')
        self.labels = np.load(self.path / '../../static/human-h9.labels.npy')
        self.description_ = (
            "Dataset of developmental data of a human (h9).  "
            f"Preprocessed and reduced to {self.data.shape[1]} "
            "dimension."
        )


class TreutleinHumanB2(TreutleinChimp):
    def load(self):
        self.data = np.load(self.path / '../../static/human-409b2.data.npy')
        self.labels = np.load(self.path / '../../static/human-409b2.labels.npy')
        self.description_ = (
            "Dataset of developmental data of a human (409b2).  "
            f"Preprocessed and reduced to {self.data.shape[1]} "
            "dimension."
        )
