from . import redo

import sys
import os
import inspect
from pathlib import Path
from multiprocessing.pool import Pool

import dcor
import numpy as np
from umap.umap_ import find_ab_params

def correlate(x, y):
    return np.sqrt(dcor.u_distance_correlation_sqr(x, y))


def tsne_from_rho(rho, dsrc):
    tsne = "affinity/stdscale;f:1e-4/tsne"
    if rho > 12:
        tsne += f";early_exaggeration:{rho:g}"

    if rho != 1:
        tsne += f";late_exaggeration:{rho:g}"

    return dsrc / tsne / "data.npy"


def do_tsne(arg):
    ref = arg["ref"]
    tsnes = arg["tsnes"]
    subsel = arg["subsel"]

    return [correlate(ref[subsel], np.load(tsne)[subsel]) for tsne in tsnes]

def correlate_dataset(dsrc, rhos, n_subsel=5000, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(555)

    min_dist = 0.1
    spread = 1.0
    a, b = find_ab_params(spread, min_dist)

    # order here is important and needs to be accounted for when using the dict
    setup = {
        "fa2": dsrc / "ann/stdscale;f:1e3/fa2/data.npy",
        "umap": dsrc / "umap_knn/maxscale;f:10/umap/data.npy",
        "fa2-ri": dsrc / "ann/random/stdscale;f:1e3/fa2/data.npy",
        "umap-default": dsrc / f"umap_knn/spectral/maxscale;f:10/umap;a:{a};b:{b}/data.npy",
    }

    tsnes = [tsne_from_rho(rho, dsrc) for rho in rhos]
    datafiles = list(setup.values()) + tsnes

    # the computation happens here
    redo.redo_ifchange(datafiles)


    n = np.load(dsrc / "data.npy").shape[0]
    subsel = random_state.choice(
        n, min(n_subsel, n), replace=False
    )
    corrs = {k: [] for k in setup.keys()}

    algo_data = [np.load(f) for f in setup.values()]
    args = [dict(ref=a, tsnes=tsnes, subsel=subsel) for a in algo_data]
    results = Pool(len(setup)).map(do_tsne, args)

    return {k: res for k, res in zip(setup.keys(), results)}


def get_rhos():
    """Create a list of 50 values spaced evenly on a log scale and add rho=4 and rho=30 for experiments."""
    rhos = np.logspace(np.log10(1), np.log10(100)).round(1)
    return sorted(list(rhos) + [4, 30])


def pca_maybe(dataname):
    needs_pca = str(dataname) in ["mnist", "famnist", "kuzmnist", "kannada"]

    return "pca" if needs_pca else "."
