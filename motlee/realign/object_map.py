import numpy as np
from dataclasses import dataclass

@dataclass
class ObjectMap():
    centroids: np.array = None
    widths: np.array = None
    heights: np.array = None
    ages: np.array = None

    def __len__(self):
        if self.centroids is None:
            return 0
        return self.centroids.shape[0]