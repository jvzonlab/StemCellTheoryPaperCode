import math
from typing import Dict, Any, List, Tuple

import numpy

from stem_cell_model.lineages import Lineages


class CloneSizeDistribution:
    """Represents a clone size distribution: how many clones are there with a given size?"""

    _clone_sizes: Dict[int, int]

    @staticmethod
    def of_single_clone(clone_size: int) -> "CloneSizeDistribution":
        """Returns a clone size "distribution" consisting of only a single clone."""
        distribution = CloneSizeDistribution()
        distribution._clone_sizes[clone_size] = 1
        return distribution

    @staticmethod
    def of_clone_sizes(*args: int)-> "CloneSizeDistribution":
        """Returns a clone size distribution of the given sizes. For example, (3, 4, 3, 5) is
        a clone size distribution where clone size 3 is the most frequent.."""
        distribution = CloneSizeDistribution()
        for clone_size in args:
            distribution.add_clone_size(clone_size)
        return distribution

    def __init__(self):
        self._clone_sizes = dict()

    def add_clone_size(self, clone_size: int):
        """Add a single clone size to this distribution."""
        if clone_size in self._clone_sizes:
            self._clone_sizes[clone_size] += 1
        else:
            self._clone_sizes[clone_size] = 1

    def merge(self, other: "CloneSizeDistribution"):
        """Adds all data from the other clone size distribution to this clone size distribution."""
        for clone_size, count in other._clone_sizes.items():
            if clone_size in self._clone_sizes:
                self._clone_sizes[clone_size] += count
            else:
                self._clone_sizes[clone_size] = count

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CloneSizeDistribution):
            return False
        return other._clone_sizes == self._clone_sizes

    def __repr__(self) -> str:
        return "CloneSizeDistribution(" + repr(self._clone_sizes) + ")"

    def max(self) -> int:
        """Gets the highest occuring clone size."""
        return max(self._clone_sizes.keys())

    def to_flat_array(self) -> numpy.ndarray:
        """Returns a flat array of the clone sizes. If this distribution has 5 occurrences of clone size
        3 and 2 occurrences of clone size 4, then this returns [3, 3, 3, 3, 3, 4, 4].

        Of course, this is quite wasteful for RAM. This method mainly exists for compatibility with
        old scripts that use matplotlib.pyplot.hist. You should instead use a bar plot with
        indices() and to_height_array()."""
        length = sum(self._clone_sizes.values())
        array = numpy.empty(length, dtype=numpy.uint16)

        i = 0
        for clone_size, count in self._clone_sizes.items():
            for _ in range(count):
                array[i] = clone_size
                i += 1
        return array

    def get_clone_size_count(self, clone_size: int) -> int:
        """Gets how many times the given clone size was found."""
        clone_size_count = self._clone_sizes.get(clone_size)
        if clone_size_count is None:
            return 0
        return clone_size_count

    def indices(self) -> List[int]:
        """Returns [1, 2, 3, ..., self.max()]."""
        return list(range(1, self.max() + 1))

    def to_height_array(self) -> List[int]:
        """Gets how often each clone size occurs, starting from clone size 1 (at position 0)."""
        return_values = list()
        for i in range(1, self.max() + 1):
            return_values.append(self.get_clone_size_count(i))
        return return_values

    def get_average_and_st_dev(self) -> Tuple[float, float]:
        total_size = sum(clone_size * count for clone_size, count in self._clone_sizes.items())
        total_count = sum(self._clone_sizes.values())
        average = total_size / total_count

        variance = 1 / (total_count - 1) * sum((clone_size ** 2) * count for clone_size, count in self._clone_sizes.items()) - (total_count / (total_count - 1)) * average ** 2

        return average, math.sqrt(variance)


def get_clone_size_distribution(lineage: Lineages, min_time: float, max_time: float) -> CloneSizeDistribution:
    """Gets the clone size distribution of this lineage tree. For each cell that exists at min_time, the clone size
    at max_time is returned."""
    distribution = CloneSizeDistribution()
    for track in lineage.get_tracks():
        if track.exists_at_time(min_time):
            distribution.add_clone_size(track.get_clone_size(max_time))
    return distribution


def get_clone_size_distributions_with_duration(lineage: Lineages, min_time: float, max_time: float, duration: float,
                                               increment: int = 5) -> CloneSizeDistribution:
    """# gets the clone size distributions of the given duration for this lineage tree. If min_time is 0, max_time is
    70, duration is 50 and increment is 5, then this will return the clone sizes for [0, 50], [5, 55], [10, 60],
    [15, 65] and [20, 70]."""
    clone_sizes = CloneSizeDistribution()
    for start_time in range(int(min_time), int(max_time - duration + 1), increment):
        clone_sizes.merge(get_clone_size_distribution(lineage, start_time, start_time + duration))
    return clone_sizes
