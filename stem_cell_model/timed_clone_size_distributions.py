from typing import List

from stem_cell_model.clone_size_distributions import CloneSizeDistribution
from stem_cell_model.lineages import Lineages


class TimedCloneSizeDistribution:
    """Shows how a clone size distribution evolves over time."""
    _interval: float
    _distributions: List[CloneSizeDistribution]

    def __init__(self, interval: float, distributions: List[CloneSizeDistribution]):
        self._interval = interval
        self._distributions = distributions

    def get_times(self):
        return [i * self._interval for i in range(1, len(self._distributions) + 1)]

    def get_clone_size_counts(self, min_clone_size: int, max_clone_size: int) -> List[int]:
        """Gets the clone size count over time for the given clone sizes. For the times corresponding to the list
        indices, see get_times()."""
        clone_size_counts = list()
        for distribution in self._distributions:
            clone_size_count = 0
            for clone_size in range(min_clone_size, max_clone_size + 1):
                clone_size_count += distribution.get_clone_size_count(clone_size)
            clone_size_counts.append(clone_size_count)
        return clone_size_counts

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


def get_proliferative_clone_size_distribution(lineages: Lineages, min_time: float, max_time: float, interval: float,
                                              ) -> TimedCloneSizeDistribution:
    """Gets the proliferative clone size distribution of this lineage tree. For homeostatic systems, this distribution
    shouldn't grow forever, unlike the normal clone size distribution."""
    current_max_time = min_time + interval
    distributions = list()
    while current_max_time < max_time:
        distribution = CloneSizeDistribution()
        for track in lineages.get_tracks():
            if track.exists_at_time(min_time):
                distribution.add_clone_size(track.get_proliferative_clone_size(current_max_time))
        distributions.append(distribution)
        current_max_time += interval

    return TimedCloneSizeDistribution(interval, distributions)
