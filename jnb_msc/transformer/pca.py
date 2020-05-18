from . import NDStage

from sklearn.decomposition import PCA as SklearnPCA


class PCA(NDStage):
    def __init__(
        self,
        path,
        dataname="data.npy",
        n_components=50,
        random_state=None,
        copy=False,
        **kwargs
    ):
        super().__init__(
            path,
            dataname=dataname,
            n_components=n_components,
            random_state=random_state,
        )
        self.copy = copy
        self.kwargs = kwargs

    def transform(self):
        # n_components must not be bigger than the original features
        self.n_components = min(self.data.shape[1], self.n_components)
        self.pca = SklearnPCA(
            n_components=self.n_components,
            random_state=self.random_state,
            copy=self.copy,
            **self.kwargs
        )

        self.data_ = self.pca.fit_transform(self.data)
        return self.data_
