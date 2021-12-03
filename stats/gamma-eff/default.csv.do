#!/usr/bin/env python

# create the scale for a given dataset.  This do rule
# is meant to be invoked as `redo stats/dist-corr/mnist.csv` to get
# the distance correlations for the mnist dataset, for example.

# only works for mnist-like data as we assume that n = 70000.

# import minilib as lib
import jnb_msc.statsutil as util
import jnb_msc.redo as redo

import sys
import numpy as np
import pandas as pd

from pathlib import Path


def learning_rate(gamma):
    return min(1, gamma / 1000)


if __name__ == "__main__":
    dataroot = Path("../../data")
    dataname = Path(sys.argv[2]).with_suffix("")
    n = 70000
    eps = 0.001

    # get the dataset name via the file name
    dsrc = dataroot / dataname / util.pca_maybe(dataname)
    n_subsamples = list(range(10000, n + 1, 5000))
    gammas = np.array(
        np.logspace(np.log10(100), np.log10(100000), 40).round(),
        dtype="int",
    )
    dsrcs = [
        (dataroot / dataname / (f"subsample;n:{n_sub}" if n_sub != n else ".") / util.pca_maybe(dataname) /
         "umap_knn/maxscale;f:10/")
        for n_sub in n_subsamples
    ]
    rng = np.random.RandomState(72070)

    util.redo.redo_ifchange([util.__file__])

    tuples = []
    dirs = []
    for dsrc, n_sub in zip(dsrcs, n_subsamples):
        tuples.append((n_sub, "umap", 1))
        dirs.append(dsrc / "umap")
        for gamma in gammas:
            lr = learning_rate(gamma)

            dirs.append(
                dsrc /
                f"tsnee;early_exaggeration:1;elastic_const:{gamma:g};learning_rate:{lr};eps:{eps}"
            )
            tuples.append((n_sub, "umapbh", gamma))
            # dirs.append(dsrc / "umapbh")

    files = [d / "data.npy" for d in dirs]
    redo.redo_ifchange(files)

    ix = pd.MultiIndex.from_tuples(tuples, names=["n", "algo", "gamma"])
    df = pd.DataFrame(index=ix)

    for tup, file in zip(tuples, files):
        ar = np.load(file)
        extent = ar.max() - ar.min()
        df.loc[tup, "extent"] = extent

    if df.isna().any().bool():
        from warnings import warn

        warn("Resulting df has NaN values, please check!")

    df.to_csv(sys.argv[3])
