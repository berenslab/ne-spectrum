#!/usr/bin/env python
# coding: utf-8

# # Treutlein lab ape organoid data
# The data are here: https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-7552/
#
# Download files 1 to 7
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.1.zip
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.2.zip
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.3.zip
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.4.zip
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.5.zip
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.6.zip
# https://www.ebi.ac.uk/arrayexpress/files/E-MTAB-7552/E-MTAB-7552.processed.7.zip
# and unpack
#
# Download supplementary informationfrom the Nature paper https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-019-1654-9/MediaObjects/41586_2019_1654_MOESM3_ESM.zip, extract `Supplementary_Table_5.txt` and rename it into `metadata_chimp_cells_suppl.tsv`. The metadata file on arrayexpress seems to be wrong. I wrote to the authors to clarify.

import numpy as np
import pandas as pd
import matplotlib
import scipy
import openTSNE

from scipy import sparse
from scipy.io import mmread
from pathlib import Path

import rnaseqTools


def preprocess(metafile, countfile, line, n=1000, decay=1.5, n_components=50):
    meta = pd.read_csv(metafile, sep="\t")

    counts = mmread(str(countfile))
    counts = scipy.sparse.csc_matrix(counts).T

    ind = meta["in_FullLineage"].values
    if line is not None:
        ind = ind & (meta["Line"].values == line)

    seqDepths = np.array(counts[ind, :].sum(axis=1))
    stage = meta["Stage"].values[ind].astype("str")

    impGenes = rnaseqTools.geneSelection(counts[ind, :], n=n, decay=decay, plot=False)

    # Transformations

    X = np.log2(counts[:, impGenes][ind, :] / seqDepths * np.median(seqDepths) + 1)
    X = np.array(X)
    X = X - X.mean(axis=0)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :n_components]

    return X, stage


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Preprocess developmental data.  This requires the data to be "
        "downloaded and unzipped already, prior to running this.  Read "
        "the comments at the top of this file in order to get the links "
        " that point to the data files and some further instructions."
    )
    parser.add_argument(
        "--type",
        default="chimp",
        help="Whether chimp or human data should be extracted",
    )
    parser.add_argument("--line", default=None, help="Which line should be saved")
    parser.add_argument(
        "--prefix",
        type=Path,
        default="~/Downloads/treutlein",
        help="The location of the raw data",
    )
    parser.add_argument(
        "--outfmt", default="npy", help="Whether to output npy or pickle files"
    )

    args = parser.parse_args()

    # select a default line for human
    if args.type == "human" and (args.line is None):
        args.line = "409b2"

    if args.type == "chimp":
        metafile = "metadata_chimp_cells_suppl.tsv"
        countfile = "chimp_cell_counts_consensus.mtx"
        line = None
    elif args.type == "human":
        metafile = "metadata_human_cells.tsv"
        countfile = "human_cell_counts_consensus.mtx"
        line = args.line

    if line not in [None, "409b2", "H9"]:
        raise ValueError(f"Value for line {line} is not valid.")

    metafile = args.prefix.expanduser() / metafile
    countfile = args.prefix.expanduser() / countfile
    X, stage = preprocess(metafile, countfile, line)

    outputfile = "{}{}".format(args.type, "" if args.line is None else f"-{args.line}")

    if args.outfmt == "pickle":
        import pickle

        pickle.dump([X, stage], open(outputfile + ".pickle", "wb"))
    elif args.outfmt == "npy":
        np.save(outputfile + ".data.npy", X)
        np.save(outputfile + ".labels.npy", stage)
