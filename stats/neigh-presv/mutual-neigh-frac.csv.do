#!/usr/bin/env python

import jnb_msc.statsutil as util

import sys
import numpy as np
import pandas as pd

from pathlib import Path

if __name__ == "__main__":
    datasets = [
        "mnist",
        "famnist",
        # "kuzmnist",
        # "kannada",
        # "treutlein",
        # "treutlein_h9",
        # "treutlein_409b2",
        # "tasic",
        # "hydra",
        # "zfish",
        # "gauss_devel",
    ]
    dataroot = Path("../../data")

    rng = np.random.RandomState(72070)

    csvs = [Path(ds).with_suffix(".csv") for ds in datasets]
    # computation
    util.redo.redo_ifchange([util.__file__] + csvs)

    rhos = util.get_rhos()
    rhos = [1, 4, 30]

    ix = pd.Index(rhos, name="rho")
    df = pd.DataFrame(index=ix)
    for dataset, csv in zip(datasets, csvs):
        df1 = pd.read_csv(csv, index_col=["rho"])
        df[dataset] = df1["mutual-neigh-frac"]


    df.to_csv(sys.argv[3])
