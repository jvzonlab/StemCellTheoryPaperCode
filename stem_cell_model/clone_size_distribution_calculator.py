from typing import Callable

from numpy.random import Generator

from stem_cell_model.lineages import CloneSizeDistribution
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.results import SimulationResults

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
        for lineage in results.lineages:
            clone_size_distribution.merge(lineage.get_clone_size_distribution(
                clone_size_config.t_wait, clone_size_config.t_wait + clone_size_config.t_clone_size))
    return clone_size_distribution

