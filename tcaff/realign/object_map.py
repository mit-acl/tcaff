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
        widths = self.widths.reshape(-1).tolist() if self.widths is not None else [None for _ in range(self._n)]
        heights = self.heights.reshape(-1).tolist() if self.heights is not None else [None for _ in range(self._n)]
        ages = self.ages.reshape(-1).tolist() if self.ages is not None else [None for _ in range(self._n)]
        yield from [Object(np.array(centroid), width, height, age) for (centroid, width, height, age) \
                    in zip(centroids, widths, heights, ages)]
    
    def plot2d(self, ax=None, max_obj_width=np.inf, circles=True, centroids=False, **kwargs):
        if ax is None:
            ax = plt.gca()
        for object in self:
            if object.width > max_obj_width: continue
            if circles:
                circ = plt.Circle(object.centroid[:2], object.width, fill=False, **kwargs)
                ax.add_artist(circ)
            if centroids:
                ax.scatter(*object.centroid[:2], **kwargs)
        return ax
    
    def as_array(self):
        l = []
        for object in self:
            assert object.centroid is not None
            new_item = object.centroid.reshape(-1).tolist()
            if object.width is not None:
                new_item += [object.width]
            if object.height is not None:
                new_item += [object.height]
            if object.age is not None:
                new_item += [object.age]
            l.append(new_item)
        return np.array(l)
    
    def __add__(self, other):
        assert (self.centroids is not None and other.centroids is not None) or \
                (self.centroids is None and other.centroids is None), "Maps must use same fields"
        assert (self.widths is not None and other.widths is not None) or \
                (self.widths is None and other.widths is None), "Maps must use same fields"
        assert (self.heights is not None and other.heights is not None) or \
                (self.heights is None and other.heights is None), "Maps must use same fields"
        assert (self.ages is not None and other.ages is not None) or \
                (self.ages is None and other.ages is None), "Maps must use same fields"
        if self.centroids is None:
            centroids = None
        else:
            centroids = np.concatenate([self.centroids, other.centroids], axis=0)
        if self.widths is None:
            widths = None
        else:
            widths = np.concatenate([self.widths, other.widths], axis=0)
        if self.heights is None:
            heights = None
        else:
            heights = np.concatenate([self.heights, other.heights], axis=0)
        if self.ages is None:
            ages = None
        else:
            ages = np.concatenate([self.ages, other.ages], axis=0)
        new_map = ObjectMap(centroids=centroids, widths=widths, heights=heights, ages=ages)
        return new_map

@dataclass
class Object():
    centroid: np.array = None
    width: float = None
    height: float = None
    age: float = None