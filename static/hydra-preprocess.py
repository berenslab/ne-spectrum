#!/usr/bin/env python
# coding: utf-8

# Hydra polyp developmental data from
# https://science.sciencemag.org/content/365/6451/eaav9314.long.

# Download the data from
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121617 and
# have the prefix argument point to the folder that contains it.  Note
# that this script operates directly on the gzipped files.

import numpy as np
import pandas as pd
import gzip
import sys

from scipy.io import mmread
from pathlib import Path

import rnaseqTools


def preprocess(tfile, lfile, chunksize=10000):
    with gzip.open(tfile) as f:
        head = f.readline()
        columns = ["cell"] + [str(b, encoding="utf8")[1:-1] for b in head.split()]
        counts, genes, cells = rnaseqTools.sparseload(
            f, names=columns, chunksize=chunksize, sep="\t"
        )

    meta = pd.read_csv(lfile)
    # create shorthands to make the listcomp more readable
    ns = meta["NAME"].values
    cs = meta["Cluster"]
    labelled_cells = np.isin(cells, ns)

    # subselect cells in labels and counts
    cells = cells[labelled_cells]
    counts = counts[labelled_cells]

    def g2c(c):
        if c.startswith("en"):
            return "xkcd:light green"
        elif c.startswith("ec"):
            return "xkcd:sky blue"
        elif c.startswith("i_"):
            return "xkcd:salmon"
        else:
            return "xkcd:dark grey"

    # transform group assignments into colors
    stage = [g2c(cs[ns == c].item()) for c in cells]
    stage = np.array(stage, dtype="str")

    important_genes = rnaseqTools.geneSelection(
        counts, n=1000, genes=genes, plot=False,
    )

    librarySizes = np.sum(counts, axis=1)
    X = np.log2(counts[:, important_genes] / librarySizes * 1e6 + 1)
    X = np.array(X)
    X = X - X.mean(axis=0)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :50]

    return X, stage


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
    transcriptome_count = "GSE121617_Hydra_DS_transcriptome_UMICounts.txt.gz"
    args = parser.parse_args()

    p = args.prefix.expanduser()
    metafile = opath / "metadata-hydra.csv"
    X, stage = preprocess(p / transcriptome_count, metafile)

    outputfile = "hydra"

    if args.outfmt == "pickle":
        import pickle

        pickle.dump([X, stage], open(opath / (outputfile + ".pickle"), "wb"))
    elif args.outfmt == "npy":
        np.save(opath / (outputfile + ".data.npy"), X)
        np.save(opath / (outputfile + ".labels.npy"), stage)
