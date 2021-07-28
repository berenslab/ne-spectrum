#!/usr/bin/env python

import jnb_msc.statsutil as lib

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
    dataroot = Path("../../data")
    dataname = Path(sys.argv[2]).with_suffix("")

    # get the dataset name via the file name
    dsrc = dataroot / dataname / lib.pca_maybe(dataname)
    rng = np.random.RandomState(72070)

    rhos = lib.get_rhos()

    tsnes = [lib.tsne_from_rho(rho, dsrc) for rho in rhos]
    lib.redo.redo_ifchange([lib.__file__] + tsnes)

    zns = np.array([computeZ(np.load(X), m=1000) for X in tsnes])
    n = np.load(dsrc / "data.npy").shape[0]
    # lognz = np.log(-zs)
    nrhozs = [(1 / zn) / rho for rho, zn in zip(rhos, zns)]
    df = pd.DataFrame({"n/rho*Z": nrhozs, "Z/n": zns}, index=pd.Index(rhos, name="rho"))

    df.to_csv(sys.argv[3])
