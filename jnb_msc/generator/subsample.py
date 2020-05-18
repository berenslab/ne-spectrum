from sklearn.datasets import fetch_openml
import numpy as np

from . import GenStage


class Subsample(GenStage):
    def __init__(
        self,
        path,
        dataname="data.npy",
        labelname="labels.npy",
        random_state=None,
        n=1000,
    ):
        super().__init__(
            path, dataname=dataname, labelname=labelname, random_state=random_state
        )
        self.n = n

    def get_datadeps(self):
        return [
            self.indir / self.dataname,
            self.indir / self.labelname,
            self.indir / self.descname,
        ]

    def load(self):
        inlist = self.get_datadeps()
        self.data = np.load(inlist[0])
        self.labels = np.load(inlist[1])
        self.description = inlist[2].read_text()

    def transform(self):
        self.indices = self.random_state.choice(self.data.shape[0], size=self.n)
        self.data_ = self.data[self.indices]
        self.labels_ = self.labels[self.indices]
        self.description_ = self.description + f"  Downsampled to {self.n} samples."

        return self.data_
