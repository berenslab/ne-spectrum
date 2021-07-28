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
        "kuzmnist",
        "kannada",
        "treutlein",
        "treutlein_h9",
        "treutlein_409b2",
        "tasic",
        "hydra",
        "zfish",
        "gauss_devel",
    ]
    dataroot = Path("../../data")

    rng = np.random.RandomState(72070)

    csvs = [Path(ds).with_suffix(".csv") for ds in datasets]
    # computation
    util.redo.redo_ifchange([util.__file__] + csvs)

    rhos = util.get_rhos()

    df1 = pd.read_csv(csvs[0], index_col=["algo", "rho"])
    ix = df1.index
    df = pd.DataFrame(index=ix)
    for dataset, csv in zip(datasets, csvs):
        df1 = pd.read_csv(csv, index_col=ix.names)
        df[dataset] = df1["corr"]


    df.to_csv(sys.argv[3])
