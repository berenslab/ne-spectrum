from . import redo

import sys
import os
import inspect

import dcor
import numpy as np

from pathlib import Path


def correlate(x, y):
    return np.sqrt(dcor.u_distance_correlation_sqr(x, y))


def tsne_from_rho(rho, dsrc):
    tsne = "affinity/stdscale;f:1e-4/tsne"
    if rho > 12:
        tsne += f";early_exaggeration:{rho:g}"

    if rho != 1:
        tsne += f";late_exaggeration:{rho:g}"

    return dsrc / tsne / "data.npy"


def correlate_dataset(dsrc, rhos, n_subsel=5000, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(555)
    fa2 = dsrc / "ann/stdscale;f:1e3/fa2/data.npy"
    umap = dsrc / "umap_knn/maxscale;f:10/umap/data.npy"

    tsnes = [tsne_from_rho(rho, dsrc) for rho in rhos]
    datafiles = [fa2, umap] + tsnes

    # the computation happens here
    redo.redo_ifchange(datafiles)

    fa2, umap = np.load(fa2), np.load(umap)

    subsel = random_state.choice(
        fa2.shape[0], min(n_subsel, fa2.shape[0]), replace=False
    )
    corr_fa2 = []
    corr_umap = []
    for tsne_f in tsnes:
        tsne = np.load(tsne_f)
        corr_fa2.append(correlate(fa2[subsel], tsne[subsel]))
        corr_umap.append(correlate(umap[subsel], tsne[subsel]))

    return corr_fa2, corr_umap


def get_rhos():
    """Create a list of 50 values spaced evenly on a log scale and add rho=4 and rho=30 for experiments."""
    rhos = np.logspace(np.log10(1), np.log10(100)).round(1)
    return sorted(list(rhos) + [4, 30])


def pca_maybe(dataname):
    needs_pca = str(dataname) in ["mnist", "famnist", "kuzmnist", "kannada"]

    return "pca" if needs_pca else "."
