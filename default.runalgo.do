#!/usr/bin/env python
#!/home/jnb/.local/miniconda3/bin/python3
# -*- mode: python-mode -*-

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
    name = Path(sys.argv[2]).parent
    algo = jnb_msc.from_string(name)

    filedeps = set(
        [
            mod.__file__
            for mod in [inspect.getmodule(m) for m in algo.__class__.mro()]
            if hasattr(mod, "__file__")  # and isinstance(mod, jnb_msc.abpc.ProjectBase)
        ]
    )

    datadeps = algo.get_datadeps()
    redo.redo_ifchange(list(filedeps) + datadeps)

    algo()
