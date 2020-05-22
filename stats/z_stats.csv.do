#!/usr/bin/env python

import minilib as lib

import sys
import numpy as np
import pandas as pd

from pathlib import Path
from numba import njit


@njit
def computeZ(Z, m=1000):
    s = 0
    for i in range(m):
        s += np.sum(1 / (1 + np.sum((Z[i, :] - Z) ** 2, axis=1))) - 1
    s /= m
    return s


if __name__ == "__main__":
    dsrc = Path("../data/mnist/pca")
    rng = np.random.RandomState(72070)

    rhos = lib.get_rhos()

    tsnes = [lib.tsne_from_rho(rho, dsrc) for rho in rhos]
    lib.redo.redo_ifchange(tsnes)

    zs = np.array([computeZ(np.load(X)) for X in tsnes])
    n = np.load(dsrc / "data.npy").shape[0]
    lognz = np.log(-zs)
    df = pd.DataFrame({"rho": rhos, "Z/N": zs, "log(N/Z)": lognz})

    df.to_csv(sys.argv[3])
