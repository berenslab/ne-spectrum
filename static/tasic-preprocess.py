#!/usr/bin/env python
# coding: utf-8

# Tasic et al. 2018 cell data from an adult mouse cortex.

# VISp: http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985
# ALM: http://celltypes.brain-map.org/api/v2/well_known_file_download/694413179

# To get the information about cluster colors and labels
# (tasic-cluster-info.csv), open the interactive data browser
# http://celltypes.brain-map.org/rnaseq/mouse/v1-alm, go to "Sample
# Heatmaps", click "Build Plot!" and then "Download data as CSV".


import numpy as np
import pandas as pd
import matplotlib
import scipy

from scipy import sparse
from scipy.io import mmread
from pathlib import Path

import rnaseqTools


def preprocess(VIsp_file, ALM_file, VIsp_rows, cluster_info, chunksize=10000):
    counts1, genes1, cells1 = rnaseqTools.sparseload(VIsp_file, chunksize=chunksize)
    counts2, genes2, cells2 = rnaseqTools.sparseload(ALM_file, chunksize=chunksize)

    counts = sparse.vstack((counts1, counts2), format="csc")

    cells = np.concatenate((cells1, cells2))

    if np.all(genes1 == genes2):
        genes = np.copy(genes1)

    genesDF = pd.read_csv(VIsp_rows)
    ids = genesDF["gene_entrez_id"].tolist()
    symbols = genesDF["gene_symbol"].tolist()
    id2symbol = dict(zip(ids, symbols))
    genes = np.array([id2symbol[g] for g in genes])

    clusterInfo = pd.read_csv(cluster_info)
    goodCells = clusterInfo["sample_name"].values
    ids = clusterInfo["cluster_id"].values
    labels = clusterInfo["cluster_label"].values
    colors = clusterInfo["cluster_color"].values

    clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids))])
    clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids))])
    clusters = np.copy(ids)

    ind = np.array([np.where(cells == c)[0][0] for c in goodCells])
    counts = counts[ind, :]

    areas = (ind < cells1.size).astype(int)

    clusters = clusters - 1

    tasic2018 = {
        "counts": counts,
        "genes": genes,
        "clusters": clusters,
        "areas": areas,
        "clusterColors": clusterColors,
        "clusterNames": clusterNames,
    }
    markerGenes = [
        "Snap25",
        "Gad1",
        "Slc17a7",
        "Pvalb",
        "Sst",
        "Vip",
        "Aqp4",
        "Mog",
        "Itgam",
        "Pdgfra",
        "Flt1",
        "Bgn",
        "Rorb",
        "Foxp2",
    ]

    importantGenesTasic2018 = rnaseqTools.geneSelection(
        tasic2018["counts"],
        n=3000,
        threshold=32,
        markers=markerGenes,
        genes=tasic2018["genes"],
        plot=False,
    )

    librarySizes = np.sum(tasic2018["counts"], axis=1)
    X = np.log2(
        tasic2018["counts"][:, importantGenesTasic2018] / librarySizes * 1e6 + 1
    )
    X = np.array(X)
    X = X - X.mean(axis=0)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :50]

    # not sure what the indexing into the labels does exactly
    return X, tasic2018["clusterColors"][tasic2018["clusters"]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Preprocess developmental data.  This requires the data to be "
        "downloaded and unzipped already, prior to running this.  Read "
        "the comments at the top of this file in order to get the links "
        "that point to the data files and some further instructions."
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default="~/Downloads/tasic",
        help="The location of the raw data",
    )
    parser.add_argument(
        "--outfmt", default="npy", help="Whether to output npy or pickle files"
    )

    args = parser.parse_args()

    p = args.prefix.expanduser()
    visp = p / "mouse_VISp_2018-06-14_exon-matrix.csv"
    alm = p / "mouse_ALM_2018-06-14_exon-matrix.csv"
    visp_rows = p / "mouse_VISp_2018-06-14_genes-rows.csv"
    cluster_info = Path() / "tasic-cluster-info.csv"
    X, stage = preprocess(visp, alm, visp_rows, cluster_info)

    outputfile = "tasic"

    if args.outfmt == "pickle":
        import pickle

        pickle.dump([X, stage], open(outputfile + ".pickle", "wb"))
    elif args.outfmt == "npy":
        np.save(outputfile + ".data.npy", X)
        np.save(outputfile + ".labels.npy", stage)
