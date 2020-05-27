from . import SimStage
from ..util import from_string

import numpy as np

from openTSNE import TSNEEmbedding
from openTSNE.callbacks import Callback as TSNECallback
from openTSNE.affinity import Affinities
from pathlib import Path


class TSNESaveEmbedding(TSNECallback):
    def __init__(self, path_prefix):
        self.path_prefix = Path(path_prefix)
        self.counter = 0
        self.path_prefix.mkdir(parents=True, exist_ok=True)

    def __call__(self, iteration, error, embedding):
        np.save(self.path_prefix / str(self.counter), embedding)
        self.counter += 1


class TSNEStage(SimStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        learning_rate="auto",
        n_iter=500,
        exaggeration=1,
        momentum=0.5,
        n_jobs=-1,
        save_iter_freq=25,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
        )
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.exaggeration = exaggeration
        self.momentum = momentum

        self.n_jobs = n_jobs
        self.saver = TSNESaveEmbedding(self.outdir)
        self.save_iter_freq = save_iter_freq

    def transform(self):
        affinities = Affinities()
        affinities.P = self.data.tocsr()

        if self.learning_rate == "auto":
            n = self.init.shape[0]
            self.learning_rate = n / self.exaggeration

        tsne = TSNEEmbedding(
            embedding=self.init,
            affinities=affinities,
            negative_gradient_method="fft",
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            callbacks_every_iters=self.save_iter_freq,
            callbacks=self.saver,
        )

        self.data_ = tsne.optimize(
            n_iter=self.n_iter,
            exaggeration=self.exaggeration,
            momentum=self.momentum,
            inplace=True,
            n_jobs=self.n_jobs,
            propagate_exception=True,
        )

        return self.data_


class TSNE(SimStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        learning_rate="auto",
        early_n_iter=250,
        late_n_iter=500,
        early_exaggeration=12,
        late_exaggeration=1,
        early_momentum=0.5,
        late_momentum=0.8,
        n_jobs=-1,
        save_iter_freq=25,
    ):
        self.early = TSNEStage(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
            learning_rate=learning_rate,
            n_iter=early_n_iter,
            exaggeration=early_exaggeration,
            momentum=early_momentum,
            n_jobs=n_jobs,
            save_iter_freq=save_iter_freq,
        )

        self.late = TSNEStage(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
            learning_rate=learning_rate,
            n_iter=late_n_iter,
            exaggeration=late_exaggeration,
            momentum=late_momentum,
            n_jobs=n_jobs,
            save_iter_freq=save_iter_freq,
        )

    def get_datadeps(self):
        return self.early.get_datadeps()

    def load(self):
        self.early.load()

    def transform(self):
        self.early.transform()

        self.late.learning_rate = self.early.learning_rate
        self.late.data = self.early.data
        self.late.init = self.early.data_
        self.late.saver = self.early.saver

        self.late.transform()

    def save(self):
        self.late.save()
