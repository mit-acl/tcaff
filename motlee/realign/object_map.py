import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

@dataclass
class ObjectMap():
    centroids: np.array = None
    widths: np.array = None
    heights: np.array = None
    ages: np.array = None

    def __post_init__(self):
        if self.centroids is None:
            self._n = 0
        else:
            self._n = self.centroids.shape[0]

    def __len__(self):
        return self._n
    
    def __iter__(self):
        centroids = self.centroids.tolist() if self.centroids is not None else [None for _ in range(self._n)]
        widths = self.widths.tolist() if self.widths is not None else [None for _ in range(self._n)]
        heights = self.heights.tolist() if self.heights is not None else [None for _ in range(self._n)]
        ages = self.ages.tolist() if self.ages is not None else [None for _ in range(self._n)]
        yield from [Object(centroid, width, height, age) for (centroid, width, height, age) \
                    in zip(centroids, widths, heights, ages)]
    
    def plot2d(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        for object in self:
            circ = plt.Circle(object.centroid[:2], object.width, fill=False, **kwargs)
            ax.add_artist(circ)
        return ax

@dataclass
class Object():
    centroid: np.array = None
    width: float = None
    height: float = None
    age: float = None