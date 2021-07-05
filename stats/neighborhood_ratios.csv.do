#!/usr/bin/env python

import minilib as lib

import sys
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.io import mmread
from numba import jit


def sum_same(hi_rows, hi_cols, lo_rows, lo_cols, n, samples=[]):
    c = 0
    him = np.empty(hi_rows.shape, dtype=bool)
    lom = np.empty(lo_rows.shape, dtype=bool)
    for i, ix in enumerate(samples):
        him[:] = hi_rows == ix
        lom[:] = lo_rows == ix
        hi_ns = hi_cols[him]
        lo_ns = lo_cols[lom]
        c += np.in1d(lo_ns, hi_ns, assume_unique=True).sum() / lo_ns.shape[0]
        # if i % 1000 == 0: print(i, file=sys.stderr)
        # for a in lo_ns:
        #     for b in hi_ns:
        #         if a == b:
        #             c += 1
        #             break

    return c / len(samples)


def neighbor_preservation(hi_knn, lo_knn, samples=[]):
    n1 = mmread(str(hi_knn))
    n2 = mmread(str(lo_knn))
    hi_rows, hi_cols = n1.nonzero()
    lo_rows, lo_cols = n2.nonzero()

    return sum_same(hi_rows, hi_cols, lo_rows, lo_cols, n1.shape[0], samples=samples)


if __name__ == "__main__":
    ds_names = [
        "mnist/pca",
        "famnist/pca",
        "kannada/pca",
        "kuzmnist/pca",
        "treutlein",
        "treutlein_409b2",
        "hydra",
        "tasic",
        "zfish",
    ]
    dsrcs = {dname: Path("../data") / dname for dname in ds_names}
    rng = np.random.RandomState(12103)

    rhos = lib.get_rhos()
    df = pd.DataFrame(index=pd.Index(rhos, name="rho"))


    # recreate all files in one go
    datadeps = []
    for dname, dsrc in dsrcs.items():

        tsnes = [lib.tsne_from_rho(rho, dsrc) for rho in rhos]
        datadeps += [tsne_path.parent / "exact_nn/nns.mtx" for tsne_path in tsnes]
    lib.redo.redo_ifchange(datadeps)

    for dname, dsrc in dsrcs.items():
        n = np.load(dsrc / "data.npy").shape[0]
        subsamples = rng.choice(n, replace=False, size=min(10000, n))
        tsnes = [lib.tsne_from_rho(rho, dsrc) for rho in rhos]
        ps = []
        for tsne in tsnes:
            orig_knn = tsne.parent.parent.parent / "nns.mtx"
            lo_knn = tsne.parent / "exact_nn/nns.mtx"
            p = neighbor_preservation(orig_knn, lo_knn, samples=subsamples)
            ps.append(p)

        df[dname] = ps

    with open(sys.argv[3], "w") as f:
        f.write("""# the mean neighborhood preservation, computed via the approx. NN graph that
# was used for optimization and the exact NN graph of the final embedding in 2D.
# Expressed as a function of rho and evaulated for different datasets, see the
# other columns.
""")

        df.to_csv(f)
