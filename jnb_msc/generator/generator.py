from ..abpc import ProjectBase

import numpy as np


class GenStage(ProjectBase):
    def __init__(
        self,
        path,
        dataname="data.npy",
        labelname="labels.npy",
        descname="descr.md",
        random_state=None,
    ):
        super().__init__(path, random_state=random_state)
        self.dataname = dataname
        self.labelname = labelname
        self.descname = descname

    def get_datadeps(self):
        return []

    def load(self):
        """Do nothing, since the generator doesn't read in data.

        This is part of the ProjectBase functions so this dummy
        implementation should suffice for most generators."""
        pass

    def save(self):
        self.save_lambda(self.outdir / self.dataname, self.data_, np.save)
        self.save_lambda(self.outdir / self.labelname, self.labels_, np.save)
        write_text = lambda f, d: f.write(d)
        self.save_lambda(
            self.outdir / self.descname, self.description_, write_text, openmode="w"
        )

        # mark the top-level for the generated data.
        (self.path / "data.root").touch(exist_ok=True)
