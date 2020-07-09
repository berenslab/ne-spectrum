#!/usr/bin/env python

import minilib as lib

import sys
import numpy as np
import pandas as pd

from pathlib import Path

if __name__ == "__main__":
    dsrc = Path("../data/hydra")
    rng = np.random.RandomState(72070)

    lib.redo.redo_ifchange([lib.__file__])
    rhos = lib.get_rhos()

    c_fa2, c_umap = lib.correlate_dataset(dsrc, rhos, random_state=rng)
    df = pd.DataFrame({"rho": rhos, "fa2": c_fa2, "umap": c_umap})

    df.to_csv(sys.argv[3])
