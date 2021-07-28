#!/usr/bin/env python

# import minilib as lib
import jnb_msc.statsutil as util

import sys
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.io import mmread
from scipy import sparse
from numba import njit, prange
from multiprocessing import Pool


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
    k = 15

    n1 = mmread(str(hi_knn)).tocsr()
    n2 = mmread(str(lo_knn))
    inds, _vals = row_topk_csr(n1.data, n1.indices, n1.indptr, k=k)
    rows = np.repeat(np.arange(n1.shape[0]), k)
    data = np.array(n1[rows, inds.reshape(-1)])
    n1 = sparse.csr_matrix((data.squeeze(), (rows, inds.reshape(-1))))
    hi_rows, hi_cols = n1.nonzero()
    lo_rows, lo_cols = n2.nonzero()

    return sum_same(hi_rows, hi_cols, lo_rows, lo_cols, n1.shape[0], samples=samples)


# https://stackoverflow.com/questions/31790819/
# scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices#31800771
@njit(cache=True)
def row_topk_csr(data, indices, indptr, k=15):
    m = indptr.shape[0] - 1
    max_indices = np.zeros((m, k), dtype=indices.dtype)
    max_values = np.zeros((m, k), dtype=data.dtype)
    _inds = np.zeros((k,), dtype=np.uint32)

    for i in prange(m):
        top_inds = np.argsort(data[indptr[i] : indptr[i + 1]])[::-1][:k]
        _inds[: top_inds.shape[0]] = top_inds
        max_indices[i] = indices[indptr[i] : indptr[i + 1]][_inds]
        max_values[i] = data[indptr[i] : indptr[i + 1]][_inds]

    return max_indices, max_values


if __name__ == "__main__":
    dataroot = Path("../../data")
    dataname = Path(sys.argv[2]).with_suffix("")

    # get the dataset name via the file name
    dsrc = dataroot / dataname / util.pca_maybe(dataname)
    rng = np.random.RandomState(12103)

    util.redo.redo_ifchange([util.__file__])


    rhos = util.get_rhos()
    tuples = [("umap", 4), ("fa2", 30)] + [("tsne", r) for r in rhos]
    ix = pd.MultiIndex.from_tuples(tuples, names=["algo", "rho"])
    df = pd.DataFrame(index=ix)


    tsnes = [util.tsne_from_rho(rho, dsrc) for rho in rhos]
    others = [
        dsrc / "umap_knn/maxscale;f:10/umap/data.npy",
        dsrc / "ann/stdscale;f:1e3/fa2/data.npy",
    ]
    # the order of the runs is important and has to match the order of
    # `tuples', used for creating the index
    runs = others + tsnes
    datadeps = [run_path.parent / "exact_nn/nns.mtx" for run_path in runs]
    util.redo.redo_ifchange(datadeps)

    n = np.load(dsrc / "data.npy").shape[0]
    subsamples = rng.choice(n, replace=False, size=min(10000, n))
    ps = []
    for other in runs:
        orig_knn = other.parent.parent.parent / "nns.mtx"
        lo_knn = other.parent / "exact_nn/nns.mtx"
        p = neighbor_preservation(orig_knn, lo_knn, samples=subsamples)
        ps.append(p)

    df["mutual-neigh-frac"] = ps

    with open(sys.argv[3], "w") as f:
        df.to_csv(f)
