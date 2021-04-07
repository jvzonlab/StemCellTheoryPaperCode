from typing import Tuple, Dict, Any, Optional, List

import numpy
from numpy import ndarray

from stem_cell_model.lineages import Lineages


class MomentData:
    mean: ndarray  # <N>, <M>, <N^2>, <M^2>, <NM>
    sq: ndarray  # <N^2>, <M^2>
    prod: int  # <NM>

    def __init__(self):
        self.mean = numpy.zeros(2)
        self.sq = numpy.zeros(2)
        self.prod = 0

    def adjust_moment_data(self, dt: float, n: ndarray):
        # <N>,<M>
        self.mean += dt * n
        # <N^2>,<M^2>
        self.sq += dt * n ** 2
        # <NM>
        self.prod += dt * n[0] * n[1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "sq": self.sq,
            "prod": self.prod
        }


class RunStats:
    runs_ended_early: int
    t_end: int
    n_exploded: int

    def __init__(self, *, runs_ended_early: int, t_end: int, n_exploded: int):
        self.runs_ended_early = runs_ended_early
        self.t_end = t_end
        self.n_exploded = n_exploded

    def to_dict(self) -> Dict[str, Any]:
        # Yes, run_ended_early should be runs_ended_early. However, we want to keep the
        # format compatible with old data.
        return {
            'run_ended_early': self.runs_ended_early,
            't_end': self.t_end,
            'n_exploded': self.n_exploded
        }


class SimulationResults:
    """Holds all results of a simulation."""

    moments: MomentData
    n_vs_t: Optional[ndarray] = None
    u_vs_t: Optional[ndarray] = None
    lineages: Optional[List[Lineages]] = None
    run_stats: RunStats

    def __init__(self, *, moments: MomentData, run_stats: RunStats):
        self.moments = moments
        self.run_stats = run_stats

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary in the format used on disk."""
        return_dict = {
            "Moments": self.moments.to_dict(),
            "RunStats": self.run_stats.to_dict()
        }
        if self.n_vs_t is not None:
            return_dict["n_vs_t"] = self.n_vs_t
        if self.u_vs_t is not None:
            return_dict["u_vs_t"] = self.u_vs_t
        if self.lineages is not None:
            return_dict["Lineage"] = self.lineages
        return return_dict
