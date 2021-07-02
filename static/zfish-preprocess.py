#!/usr/bin/env python
# coding: utf-8

# Zebrafish embryo data from Wagner et al. 2018.  Download the ScanPy
# version from
# https://kleintools.hms.harvard.edu/paper_websites/wagner_zebrafish_timecourse2018/mainpage.html
# and have the prefix point at the containing folder.

# Direct file download, contains everything for the dataset
# https://kleintools.hms.harvard.edu/paper_websites/wagner_zebrafish_timecourse2018/WagnerScience2018.h5ad

import numpy as np
import pandas as pd
import gzip
import sys
import anndata

from scipy.io import mmread
from pathlib import Path

import rnaseqTools

lbl_map = {
    "4hpf": "navy",
    "6hpf": "royalblue",
    "8hpf": "skyblue",
    "10hpf": "lightgreen",
    "14hpf": "gold",
    "18hpf": "tomato",
    "24hpf": "firebrick",
    "unused": "maroon",
}

tissue_map = {
    "Pluripotent": "slategrey",
    "Epidermal": "darkgreen",
    "Endoderm": "darkorange",
    "Forebrain / Optic": "navy",
    "Hindbrain / Spinal Cord": "skyblue",
    "Neural Crest": "darkturquoise",
    "Midbrain": "royalblue",
    "Germline": "fuchsia",
    "Mesoderm": "tomato",
    "Other": "gainsboro",
    "NaN": "gainsboro",
}


def preprocess(anndatafile):
    ann = anndata.read(anndatafile)
    counts = ann.X
    genes = ann.var.index.astype("str")
    cells = ann.obs["unique_cell_id"].values.astype("str")

    important_genes = rnaseqTools.geneSelection(
        counts, n=1000, decay=1.5, genes=genes, plot=False,
    )

    librarySizes = np.sum(counts, axis=1)
    median = np.median(np.asarray(librarySizes).squeeze())
    X = np.log2(counts[:, important_genes] / librarySizes * median + 1)
    X = np.array(X)
    X = X - X.mean(axis=0)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :50]

    # map the group assignments to a color
    stage = ann.obs["TimeID"].map(lambda x: lbl_map[x]).values.astype("str")
    alt_colors = ann.obs["TissueName"].map(lambda x: tissue_map[x]).values.astype("str")
    return X, stage, alt_colors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Preprocess developmental data.  This requires the data to be "
        "downloaded prior to running this.  Read "
        "the comments at the top of this file in order to get the links "
        "that point to the data files and some further instructions."
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default="~/Downloads",
        help="The location of the raw data",
    )
    parser.add_argument(
        "--outfmt", default="npy", help="Whether to output npy or pickle files"
    )

    # outputpath, same dir that the script is in
    opath = Path(sys.argv[0]).parent
    annfile = "WagnerScience2018.h5ad"
    args = parser.parse_args()

    p = args.prefix.expanduser()
    X, stage, alt_c = preprocess(p / annfile)

    outputfile = "zfish"

    if args.outfmt == "pickle":
        import pickle

        pickle.dump([X, stage], open(opath / (outputfile + ".pickle"), "wb"))
    elif args.outfmt == "npy":
        np.save(opath / (outputfile + ".data.npy"), X)
        np.save(opath / (outputfile + ".labels.npy"), stage)
        np.save(opath / (outputfile + ".altlabels.npy"), alt_c)
