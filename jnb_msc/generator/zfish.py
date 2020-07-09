import numpy as np

from . import GenStage


class KleinZebraFish(GenStage):
    """Data of the zebra fish embryo data by Wagner et al.

    https://kleintools.hms.harvard.edu/paper_websites/wagner_zebrafish_timecourse2018/mainpage.html"""

    def load(self):
        self.data = np.load(self.path / "../../static/zfish.data.npy")
        self.labels = np.load(self.path / "../../static/zfish.labels.npy")
        self.description_ = "Dataset of a zebra fish embryo from Wagner et al. 2018."

    def transform(self):
        self.labels_ = self.labels
        self.data_ = self.data

        return self.data_
