import abc
import os

import numpy as np

from pathlib import Path
from tempfile import NamedTemporaryFile


class ProjectBase(abc.ABC):
    """The abstract base class for this project.

    The function defines the stubs for the functions that are expected
    to be implemented by the subclasses.  Additionally, the class
    defines the classmethod `from_string` that will instantiate the
    correct subclass, based on the string."""

    def __init__(
        self, path, indir=None, outdir=None, random_state=None, make_dirs=True
    ):
        self.path = Path(path)

        self.indir = self.resolve_path(self.path, indir, "..")
        self.outdir = self.resolve_path(self.path, indir, "prepped")

        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is None:
            self.random_state = np.random.RandomState(0xDEADBEEF)
        else:
            self.random_state = np.random.RandomState(random_state)

        if make_dirs:
            self.path.mkdir(parents=True, exist_ok=True)
            self.outdir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def get_datadeps(self):
        """Returns a list of files that will be loaded in self.load().

        This function is used to keep track of what files need to be
        updated in order to keep the state of the generated artifacts
        up to date."""
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self):
        """Access the file system and load the files into memory.

        Ideally this should use the variables from self.get_datadeps()
        in order to make the dependency explicit and not cause a
        difference between what's declared a dependency and what is
        actually loaded."""
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self):
        raise NotImplementedError()

    def __call__(self):
        self.load()
        d = self.transform()
        self.save()
        return d

    @staticmethod
    def save_lambda(fname, data, function, openmode="wb"):
        """Saves the file via the specified lambda function.

        The rationale for this function is that if we write to the
        files that we hardlinked to, `redo` will notice a change in
        the source file and error out.  This way we write to a
        tempfile and atomically move it to a location that we will
        hardlink to.  This way we get a new inode that is not linked
        to by the target that we want to create at a later point.

        Parameters
        ----------

        fname : str or path-like
        The name that the file will be written to in the end.
        data
        This will be passed to the parameter `function` and will
        through this be written to the disk.
        function : function or lambda
        The function that will write to the opened file.  Use
        `np.save` if you want to save a numpy array.
        mode : str
        The mode for opening the file.  Will be passed onto
        `NamedTemporaryFile` as is for opening."""

        with NamedTemporaryFile(openmode) as f:
            function(f, data)

            tmpname = "{}.{}".format(f.name, Path(fname).name)
            os.link(f.name, tmpname)
        os.replace(tmpname, fname)

    @staticmethod
    def find_upwards(path, name):
        return find_upwards(path, name)

    @staticmethod
    def resolve_path(path, name, default_name=""):
        """Piece together path and name, if it makes sense.

        This is used as a shorthand to reduce boilerplate.  If `name`
        is None then we'll use the `default_name`.  But if name is an
        absolute name we will use that instead, since we assume that
        some thought was put into that (at least the path had to be
        resolved at some point).  Otherwise just concat the path and
        the name."""
        if name is None:
            return Path(path) / default_name

        name = Path(name)
        if name.is_absolute():
            return name
        else:
            return path / name

    @staticmethod
    def lim(lo, hi, eps=0.025):
        l = abs(hi - lo)
        return lo - l * eps, hi + l * eps


def find_upwards(path, name):
    if (path / name).exists():
        return path / name
    elif path.parent != path:
        return find_upwards(path.parent, name)
    elif path.parent == path:
        raise RuntimeError("Name {} not found.".format(name))
