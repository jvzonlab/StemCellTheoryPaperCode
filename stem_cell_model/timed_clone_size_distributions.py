from typing import List, Iterable

import numpy
from numpy import ndarray

from stem_cell_model.clone_size_distributions import CloneSizeDistribution
from stem_cell_model.lineages import Lineages


class TimedCloneSizeDistribution:
    """Shows how a clone size distribution evolves over time."""
    _interval: float
    _distributions: List[CloneSizeDistribution]

    def __init__(self, interval: float, distributions: List[CloneSizeDistribution]):
        self._interval = interval
        self._distributions = distributions

    @property
    def interval(self) -> float:
        """Gets the time interval at which clone sizes are calculated."""
        return self._interval  # property to keep this variable read-only

    def get_durations(self) -> List[float]:
        return [i * self._interval for i in range(1, len(self._distributions) + 1)]

    def get_clone_size_frequency_over_time(self, clone_sizes: Iterable[int]) -> ndarray:
        """Gets sum of the the clone size frequencies over time for the given clone sizes. For the times corresponding to the list
        indices, see get_durations()."""
        clone_size_counts = list()
        for distribution in self._distributions:
            clone_size_count = 0
            for clone_size in clone_sizes:
                clone_size_count += distribution.get_clone_size_frequency(clone_size)
            clone_size_counts.append(clone_size_count)
        return numpy.array(clone_size_counts, dtype=numpy.int)

    def get_average_clone_size_over_time(self) -> ndarray:
        """Gets sum of the the clone size frequencies over time for the given clone sizes. For the times corresponding
        to the list indices, see get_durations()."""
        return numpy.array([distribution.get_average() for distribution in self._distributions], dtype=numpy.float)

    def merge(self, other: "TimedCloneSizeDistribution"):
        """Sums all data from the other clone size distribution into this one. The other distribution is not modified.

        Raises ValueError if the other distribution has a different interval or time length."""
        if other._interval != self._interval or len(other._distributions) != len(self._distributions):
            raise ValueError("Incompatible timed clone size distributions")

        for self_distribution, other_distribution in zip(self._distributions, other._distributions):
            self_distribution.merge(other_distribution)

    def last(self) -> CloneSizeDistribution:
        """Returns the distribution at the last time point."""
        return self._distributions[-1]

    def get_clone_count_over_time(self) -> ndarray:
        """Gets the number of surviving clones over time."""
        return numpy.array([distribution.get_clone_count() for distribution in self._distributions], dtype=numpy.int)

    def get_distribution_at(self, time_index: int) -> CloneSizeDistribution:
        """Gets the clone size distribution at the given time index. (Same index uas used in get_durations().)"""
        if time_index < 0:
            raise IndexError(f"Negative index {time_index} not allowed, as this is likely a bug."
                             f" (You can use self.last().)")
        return self._distributions[time_index]

    def get_clone_sizes(self) -> ndarray:
        """Returns all occuring clone sizes: [1, 2, 3, ..., max]"""
        max_clone_size = max((distribution.max() for distribution in self._distributions))
        return 1+  numpy.arange(max_clone_size, dtype=numpy.int)


def get_proliferative_clone_size_distribution(lineages: Lineages, min_time: float, max_time: float, interval: float,
                                              ) -> TimedCloneSizeDistribution:
    """Gets the proliferative clone size distribution of this lineage tree. So non-dividing cells aren't included in
    this particular distribution. For homeostatic systems, this distribution shouldn't grow forever, unlike the normal
    clone size distribution."""
    current_max_time = min_time + interval
    distributions = list()
    while current_max_time <= max_time:
        distribution = CloneSizeDistribution()
        for track in lineages.get_tracks():
            if track.exists_at_time(min_time):
                distribution.add_clone_size(track.get_proliferative_clone_size(current_max_time))
        distributions.append(distribution)
        current_max_time += interval

    return TimedCloneSizeDistribution(interval, distributions)


def get_niche_clone_size_distribution(lineages: Lineages, min_time: float, max_time: float, interval: float,
                                              ) -> TimedCloneSizeDistribution:
    """Gets the niche clone size distribution of this lineage tree. So cell outside the stem cell niche aren't included
    int this particular clone size distribution. For homeostatic systems, this distribution shouldn't grow forever,
    unlike the normal clone size distribution."""
    current_max_time = min_time + interval
    distributions = list()
    while current_max_time <= max_time:
        distribution = CloneSizeDistribution()
        for track in lineages.get_tracks():
            if track.exists_at_time(min_time) and track.compartment.get_compartment_at(min_time) == 0:
                distribution.add_clone_size(track.get_niche_clone_size(current_max_time))
        distributions.append(distribution)
        current_max_time += interval

    return TimedCloneSizeDistribution(interval, distributions)
