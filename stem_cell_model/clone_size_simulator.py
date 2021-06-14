from typing import Callable

from numpy.random import Generator

from stem_cell_model import clone_size_distributions, timed_clone_size_distributions
from stem_cell_model.clone_size_distributions import CloneSizeDistribution
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.results import SimulationResults
from stem_cell_model.timed_clone_size_distributions import TimedCloneSizeDistribution

Simulator = Callable[[SimulationConfig, SimulationParameters], SimulationResults]


class CloneSizeSimulationConfig:
    t_clone_size: int  # Recording time. The longer, the larger the clone sizes.
    n_crypts: int
    random: Generator  # Random number generator for the simulation

    def __init__(self, *, t_clone_size: int, n_crypts: int, random: Generator):
        self.t_clone_size = t_clone_size
        self.n_crypts = n_crypts
        self.random = random
        self.t_wait = 0


def calculate(simulator: Simulator, clone_size_config: CloneSizeSimulationConfig, params: SimulationParameters
              ) -> CloneSizeDistribution:
    """Calculates the resulting clone size distribution for the given parameters set."""
    config = SimulationConfig(
        t_sim=clone_size_config.t_wait + clone_size_config.t_clone_size,
        n_max=10000, random=clone_size_config.random,
        track_lineage_time_interval=(clone_size_config.t_wait, clone_size_config.t_wait + clone_size_config.t_clone_size))

    clone_size_distribution = CloneSizeDistribution()
    for i in range(clone_size_config.n_crypts):
        if i > 0 and i % 100 == 0:
            print(f"{i} crypts done...")
        results = simulator(config, params)
        clone_size_distribution.merge(clone_size_distributions.get_clone_size_distribution(results.lineages,
            clone_size_config.t_wait, clone_size_config.t_wait + clone_size_config.t_clone_size))
    return clone_size_distribution


class TimedCloneSizeSimulationConfig:
    t_clone_size: int  # Recording time.
    t_interval: int  # How often the distributions are calculated.
    n_crypts: int
    random: Generator  # Random number generator for the simulation

    def __init__(self, *, t_clone_size: int, t_interval: int, n_crypts: int, random: Generator):
        self.t_clone_size = t_clone_size
        self.t_interval = t_interval
        self.n_crypts = n_crypts
        self.random = random

    def to_niche_config(self) -> SimulationConfig:
        return SimulationConfig(
            t_sim=self.t_clone_size,
            n_max=10000, random=self.random,
            track_lineage_time_interval=(0, self.t_clone_size))


def calculate_proliferative_in_niche_over_time(simulator: Simulator, clone_size_config: TimedCloneSizeSimulationConfig,
                                               params: SimulationParameters) -> TimedCloneSizeDistribution:
    """Calculates the resulting clone size distribution for the given parameters set over time. Note that only dividing
    cells are counted here.."""
    config = clone_size_config.to_niche_config()

    total_clone_size_distribution = None
    for i in range(clone_size_config.n_crypts):
        if i > 0 and i % 100 == 0:
            print(f"{i} crypts done...")
        results = simulator(config, params)
        clone_size_distribution = timed_clone_size_distributions.get_proliferative_niche_clone_size_distribution(
                results.lineages, 0, clone_size_config.t_clone_size,
                clone_size_config.t_interval)

        if total_clone_size_distribution is None:
            total_clone_size_distribution = clone_size_distribution
        else:
            total_clone_size_distribution.merge(clone_size_distribution)

    if total_clone_size_distribution is None:
        raise ValueError("Simulated zero crypts. Check config.n_crypts")
    return total_clone_size_distribution


def calculate_niche_over_time(simulator: Simulator, clone_size_config: TimedCloneSizeSimulationConfig,
                              params: SimulationParameters) -> TimedCloneSizeDistribution:
    """Calculates the resulting clone size distribution for the given parameters set over time. Note that only dividing
    cells are counted here.."""
    config = clone_size_config.to_niche_config()

    total_clone_size_distribution = None
    for i in range(clone_size_config.n_crypts):
        if i > 0 and i % 100 == 0:
            print(f"{i} crypts done...")
        results = simulator(config, params)
        clone_size_distribution = timed_clone_size_distributions.get_niche_clone_size_distribution(
                results.lineages, 0, clone_size_config.t_clone_size,
                clone_size_config.t_interval)

        if total_clone_size_distribution is None:
            total_clone_size_distribution = clone_size_distribution
        else:
            total_clone_size_distribution.merge(clone_size_distribution)

    if total_clone_size_distribution is None:
        raise ValueError("Simulated zero crypts. Check config.n_crypts")
    return total_clone_size_distribution
