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

    runs = []
    ns = []
    rhos = []
    for n in range(5000, 70001, 5000):
        if n != 70000:
            dsub = dsrc / f"subsample;n:{n}/pca/affinity/stdscale;f:1e-4"
        else:  # full mnist
            dsub = dsrc / "pca/affinity/stdscale;f:1e-4"
        for rho in [1, 2, 3, 4]:
            tsne_ = "tsne" + ("" if rho == 1 else f";late_exaggeration:{rho}")
            ns.append(n)
            rhos.append(rho)
            runs.append(dsub / tsne_ / "data.npy")

    tsnes = runs
    lib.redo.redo_ifchange(tsnes)

    zs = np.array([computeNoverZ(np.load(X)) for X in tsnes])
    df = pd.DataFrame({"n": ns, "rho": rhos, "Z/N": zs})

    df.to_csv(sys.argv[3])
