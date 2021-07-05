#!/usr/bin/env python

# create the distance correlation for a given dataset.  This do rule
# is meant to be invoked as `redo stats/dist-corr/mnist.csv` to get
# the distance correlations for the mnist dataset, for example.

# import minilib as lib
import jnb_msc.statsutil as util

import sys
import numpy as np
import pandas as pd

from pathlib import Path

if __name__ == "__main__":
    dataroot = Path("../../data")
    dataname = Path(sys.argv[2]).with_suffix("")

    # get the dataset name via the file name
    dsrc = dataroot / dataname / util.pca_maybe(dataname)
    rng = np.random.RandomState(72070)

    util.redo.redo_ifchange([util.__file__])
    rhos = util.get_rhos()

    ix = pd.MultiIndex.from_product([["fa2", "umap"], rhos], names=["algo", "rho"])

    # computation
    c_fa2, c_umap = util.correlate_dataset(dsrc, rhos, random_state=rng)

    df = pd.DataFrame({"corr": np.hstack([c_fa2, c_umap])}, index=ix)
    df.to_csv(sys.argv[3])
