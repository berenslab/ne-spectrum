from sklearn.datasets import fetch_openml
import numpy as np

from .mnist import MNIST


class FashionMNIST(MNIST):
    """A dataset generator for MNIST.

    The data source is the Fashion-MNIST dataset.  See
    https://github.com/zalandoresearch/fashion-mnist."""

    def load(self):
        self.mnist = fetch_openml("Fashion-MNIST")
        self.description_ = (
            f"The Fashion MNIST dataset, consisting of {self.mnist.data.shape[0]} "
            f"datapoints with {self.mnist.data.shape[1]} dimensions each."
        )

    def transform(self):
        self.data_ = self.mnist.data
        self.labels_ = np.vectorize(np.int8)(self.mnist.target)

        return self.data_


class KuzushijiMNIST(FashionMNIST):
    """MNIST-like dataset consisting of Kanjis."""

    def load(self):
        self.mnist = fetch_openml("Kuzushiji-MNIST")
        self.description_ = (
            f"The Kuzushiji MNIST dataset, consisting of {self.mnist.data.shape[0]} "
            f"datapoints with {self.mnist.data.shape[1]} dimensions each."
        )


class KannadaMNIST(MNIST):
    def get_datadeps(self):
        return [
            self.path / "../../static/kannada.data.npy",
            self.path / "../../static/kannada.labels.npy",
        ]

    def load(self):
        self.data = np.load(self.path / "../../static/kannada.data.npy")
        self.labels = np.load(self.path / "../../static/kannada.labels.npy")
        self.description_ = "The Kannada MNIST dataset."

    def transform(self):
        self.data_ = self.data
        self.labels_ = self.labels.astype(np.int8)
