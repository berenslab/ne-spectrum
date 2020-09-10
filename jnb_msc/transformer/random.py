from .transformer import NDStage

from sklearn.random_projection import GaussianRandomProjection


class RandomProjection(NDStage):
    """A random projection for dimensionality reduction.

    While the intended usage for this repository abuses the
    Johnson–Lindenstrauss lemma, it should be fine for out purposes
    since this could just as well be a random initialization.  This
    way there might still be a sliver of the underlying structure
    apparent in the layout."""

    def transform(self):
        t = GaussianRandomProjection(
            n_components=self.n_components, random_state=self.random_state
        )
        self.data_ = t.fit_transform(self.data)


class RandomUniform(NDStage):
    """A random init for other dimensionality reduction algorithms."""

    def transform(self):
        t = GaussianRandomProjection(
            n_components=self.n_components, random_state=self.random_state
        )
        self.data_ = self.random_state.uniform(
            size=(self.data.shape[0], self.n_components)
        )


class RandomGauss(NDStage):
    """A random init for other dimensionality reduction algorithms."""

    def transform(self):
        self.data_ = self.random_state.normal(
            size=(self.data.shape[0], self.n_components)
        )
