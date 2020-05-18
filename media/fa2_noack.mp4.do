#!/usr/bin/env python

import jnb_msc
import jnb_msc.redo as redo

import sys
import os
import inspect

from pathlib import Path


if __name__ == "__main__":
    ps = ["../data/cchains/pca/ann/fa2;n_iter:250/", "../data/cchains/pca/ann/noack/"]

    relname = sys.argv[2]
    # passing a relative plotname will ensure that the plot will also
    # be saved in the data dir.
    anim = jnb_msc.anim.ScatterAnimations(ps, plotname=relname)
    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in anim.__class__.mro()]
            if hasattr(mod, "__file__")  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
        ]
    )

    datadeps = anim.get_datadeps()

    redo.redo_ifchange(list(filedeps) + datadeps)

    print(anim.plotname, relname, sys.argv[3], file=sys.stderr)
    anim()

    # link to the result
    # Path(sys.argv[3]).link_to(relname)
    os.link(anim.plotname, sys.argv[3])
