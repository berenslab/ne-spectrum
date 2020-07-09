import jnb_msc as j

from pathlib import Path


def name_and_dict(path):
    s = str(Path(path).resolve().name)
    parts = s.split(";")
    name = parts.pop(0)
    kwargs = {}
    for p in parts:
        key, val = p.split(":")

        # try to convert to either an int or float, otherwise pass the
        # value on as is.
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass

        # special cases for val can be handled here before being
        # passed on to the kwarg dict.

        kwargs[key] = val

    return name, kwargs


def name_class_dict():
    d = {
        "mnist": j.MNIST,
        "subsample": j.Subsample,
        "cchains": j.ClusterChains,
        "gauss_devel": j.GaussLine,
        "treutlein": j.TreutleinChimp,
        "treutlein_h9": j.TreutleinHumanH9,
        "treutlein_409b2": j.TreutleinHumanB2,
        "dumbbell": j.Dumbbell,
        "famnist": j.FashionMNIST,
        "kuzmnist": j.KuzushijiMNIST,
        "kannada": j.KannadaMNIST,
        "tasic": j.TasicMouse,
        "hydra": j.HydraTrancriptomic,
        "zfish": j.KleinZebraFish,
        "tsne": j.TSNE,
        "pca": j.PCA,
        "fa2": j.FA2,
        "noack": j.BHARModel,
        "bhnoack": j.BHARModel,
        "naivenoack": j.AttractionRepulsionModel,
        "umap": j.UMAP,
        "umapbh": j.UMAPBH,
        "umap2": j.UMAPDefault,
        "umap_knn": j.UMAPKNN,
        "spectral": j.Spectral,
        "tsnestage": j.TSNEStage,
        "tsnee": j.TSNEElastic,
        "ann": j.ANN,
        "aann": j.AsymmetricANN,
        "knn_aff": j.KNNAffinities,
        "affinity": j.PerplexityAffinity,
        "random": j.RandomGauss,
        "rnd_prj": j.RandomProjection,
        "stdscale": j.StdScale,
        "maxscale": j.MaxScale,
        "rownorm": j.RowNormalize,
        "spscale": j.ScalarScale,
        "spinv": j.InvScale,
        "spnorm": j.AxisNormalize,
        "spsym": j.Symmetrize,
    }
    return d


def name_to_class(name):
    return name_class_dict()[name]


def from_string(path):
    """Takes a string or Path object and will construct the correct
    subclass.

    The string should lead with the keyword that should map to a known
    subclass.  All further parameters should be split with a semicolon
    and the value of the parameter should be separated with a colon.

    Example:  tsne;exag:30;n_iter:450

    """
    name, kwargs = name_and_dict(path)

    p = Path(path)
    # lookup for the name-class association
    lookup = name_class_dict()
    try:
        return lookup[name](p, **kwargs)
    except KeyError:
        raise RuntimeError(
            "name '{}' does not correspond to a known class.".format(name)
        )
