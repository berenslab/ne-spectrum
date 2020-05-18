from . import NNStage

from openTSNE.affinity import PerplexityBasedNN


class PerplexityAffinity(NNStage):
    """Calculates the affinities as `openTSNE.TSNE` does by default."""

    def __init__(
        self,
        path,
        dataname="data.npy",
        outname="nns.mtx",
        random_state=None,
        perplexity=30,
        metric="euclidean",
        n_jobs=-1,
    ):
        super().__init__(path, dataname, outname, random_state)
        self.perplexity = perplexity
        self.metric = metric
        self.n_jobs = n_jobs

    def transform(self):
        self.data_ = PerplexityBasedNN(
            self.data,
            perplexity=self.perplexity,
            metric=self.metric,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        ).P  # extract just the matrix
        return self.data_
