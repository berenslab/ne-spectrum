#!/usr/bin/env python

import minilib as lib

import sys
import numpy as np
import pandas as pd

from pathlib import Path
from numba import njit


@njit
def computeNoverZ(Z, m=1000):
    s = 0
    for i in range(m):
        s += np.sum(1 / (1 + np.sum((Z[i, :] - Z) ** 2, axis=1))) - 1
    s /= m
    return s


if __name__ == "__main__":
    dsrc = Path("../data/mnist")
    rng = np.random.RandomState(72070)

    steps_n = pd.Index(range(5000, 70001, 5000), name="n")
    steps_rho = pd.Index(np.linspace(1, 10, 19), name="rho")
    runs = []
    umaps = []
    ns = []
    rhos = []
    for n in steps_n:
        if n != 70000:
            dsrc_sub = dsrc / f"subsample;n:{n}/pca"
        elif n == 70000:
            dsrc_sub = dsrc / "pca"
        umaps.append(dsrc_sub / "umap_knn/maxscale;f:10/umap/data.npy")
        for rho in steps_rho:
            tsne_ = "tsne" + ("" if rho == 1 else f";late_exaggeration:{rho:g}")
            ns.append(n)
            rhos.append(rho)
            runs.append(dsrc_sub / "affinity/stdscale;f:1e-4" / tsne_ / "data.npy")

    tsnes = runs + [
        # dsrc / "pca/affinity/stdscale;f:1e-4/tsne/data.npy",
        # dsrc / "pca/affinity/stdscale;f:1e-4/tsne;late_exaggeration:4/data.npy",
    ]
    # ns += [70000, 70000]  # add full dataset
    # rhos += [1, 4]
    lib.redo.redo_ifchange(umaps + tsnes + [lib.__file__])

    values = np.empty((len(steps_n), len(steps_rho)))
    n_subsel = 6000
    for i, n in enumerate(steps_n):
        if n != 70000:
            dsrc_sub = dsrc / f"subsample;n:{n}/pca"
        elif n == 70000:
            dsrc_sub = dsrc / "pca"
        umap = np.load(dsrc_sub / "umap_knn/maxscale;f:10/umap/data.npy")
        subsel = rng.choice(umap.shape[0], min(n_subsel, umap.shape[0]), replace=False)

        for j, rho in enumerate(steps_rho):
            tsne_ = "tsne" + ("" if rho == 1 else f";late_exaggeration:{rho:g}")
            tsne = np.load(dsrc_sub / "affinity/stdscale;f:1e-4" / tsne_ / "data.npy")

            values[i, j] = lib.correlate(umap[subsel], tsne[subsel])

    df = pd.DataFrame(values, index=steps_n, columns=steps_rho)

    df.to_csv(sys.argv[3])
