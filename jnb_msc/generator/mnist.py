from pathlib import Path
from sklearn.datasets import fetch_openml
import numpy as np

from . import GenStage


class MNIST(GenStage):
    """A dataset generator for MNIST.

    The data source is the MNIST handwritten digit dataset."""

    def load(self):
        cachedir = Path().home() / ".cache/scikit_learn_data"
        path = str(cachedir) if cachedir.exists() else None
        self.mnist = fetch_openml("mnist_784", data_home=path)

    def transform(self):
        self.data_ = self.mnist.data
        self.labels_ = np.vectorize(np.int8)(self.mnist.target)
        self.description_ = (
            f"The MNIST handwritten dataset, consisting of {self.data_.shape[0]} "
            f"datapoints with {self.data_.shape[1]} dimensions each."
        )

        return self.data_
