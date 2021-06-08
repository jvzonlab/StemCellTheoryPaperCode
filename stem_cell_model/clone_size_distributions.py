import math
from typing import Dict, Any, List, Tuple, Union

import numpy
from numpy import ndarray

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

    def get_clone_size_frequency(self, clone_size: int) -> int:
        """Gets how many times the given clone size was found."""
        clone_size_count = self._clone_sizes.get(clone_size)
        if clone_size_count is None:
            return 0
        return clone_size_count

    def indices(self) -> ndarray:
        """Returns [0, 1, 2, 3, ..., self.max()]."""
        return numpy.arange(0, self.max() + 1, dtype=numpy.int)

    def to_height_array(self) -> List[int]:
        """Gets how often each clone size occurs, starting from clone size 1 (at position 0)."""
        return_values = list()
        for i in range(0, self.max() + 1):
            return_values.append(self.get_clone_size_frequency(i))
        return return_values

    def get_average(self) -> float:
        """Gets the average clone size."""
        total_size = sum(clone_size * count for clone_size, count in self._clone_sizes.items())
        total_count = self.get_clone_count()
        return total_size / total_count

    def get_average_and_st_dev(self) -> Tuple[float, float]:
        """Gets both the average clone size and the standard deviation."""
        total_size = sum(clone_size * count for clone_size, count in self._clone_sizes.items())
        total_count = self.get_clone_count()
        average = total_size / total_count

        variance = 1 / (total_count - 1) * sum((clone_size ** 2) * count for clone_size, count in self._clone_sizes.items()) - (total_count / (total_count - 1)) * average ** 2

        return average, math.sqrt(variance)

    def get_clone_count(self) -> int:
        """Gets how many different clones there are."""
        return sum((count for clone_size, count in self._clone_sizes.items() if clone_size > 0))

    def get_clone_size_frequencies(self, clone_sizes: Union[List[int], ndarray]) -> ndarray:
        """Looks up multiple clone size frequencies at once."""
        array = numpy.zeros(len(clone_sizes), dtype=numpy.int)
        for i, clone_size in enumerate(clone_sizes):
            clone_size_count = self._clone_sizes.get(clone_size)
            if clone_size_count is not None:
                array[i] = clone_size_count
        return array


def get_clone_size_distribution(lineages: Lineages, min_time: float, max_time: float) -> CloneSizeDistribution:
    """Gets the clone size distribution of this lineage tree. For each cell that exists at min_time, the clone size
    at max_time is returned."""
    distribution = CloneSizeDistribution()
    for track in lineages.get_tracks():
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
