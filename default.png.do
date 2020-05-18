#!/usr/bin/env python

import sys
import inspect
import subprocess

from pathlib import Path

import jnb_msc
import jnb_msc.redo as redo

if __name__ == "__main__":
    # redo will pass the target name as the second arg.  The directory
    # part is the relevant one for instantiating the object, so we
    # retrieve that via the parent attribute.
    fname_noext = Path(sys.argv[2])
    pathname = fname_noext.parent
    npyname = fname_noext.with_name(str(fname_noext.name) + ".npy").absolute()

    plotter = jnb_msc.plot.ScatterSingle(
        pathname, dataname=npyname, plotname=Path(sys.argv[3]).absolute(), format="png"
    )

    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in plotter.__class__.mro()]
            if hasattr(mod, "__file__")  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
        ]
    )

    datadeps = plotter.get_datadeps()
    redo.redo_ifchange(list(filedeps) + datadeps)

    plotter()
