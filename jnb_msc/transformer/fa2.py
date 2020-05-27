from .transformer import SimStage

import numpy as np

from fa2 import ForceAtlas2


class FA2(SimStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        n_iter=750,
        repulsion=2,
        jitter_tol=1,
        theta=1.2,
        use_degrees=True,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
        )

        self.n_iter = n_iter
        self.repulsion = repulsion
        self.jitter_tol = jitter_tol
        self.theta = theta
        self.use_degrees = use_degrees

    def transform(self):

        fa2 = ForceAtlas2(
            verbose=False,
            jitterTolerance=self.jitter_tol,
            scalingRatio=self.repulsion,
            barnesHutTheta=self.theta,
            degreeRepulsion=self.use_degrees,
        )

        saver = (lambda i, pos: np.save(self.outdir / str(i), pos),)

        self.data_ = fa2.forceatlas2(
            self.data,
            pos=self.init,
            iterations=self.n_iter,
            callbacks_every_iters=1,
            callbacks=saver,
        )
