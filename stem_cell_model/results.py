from typing import Tuple, Dict, Any, Optional, List

import numpy
from numpy import ndarray

from stem_cell_model.lineages import Lineage


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
    run_ended_early: bool
    t_end: int
    n_exploded: int

    def __init__(self, *, run_ended_early: bool, t_end: int, n_exploded: int):
        self.run_ended_early = run_ended_early
        self.t_end = t_end
        self.n_exploded = n_exploded

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_ended_early': self.run_ended_early,
            't_end': self.t_end,
            'n_exploded': self.n_exploded
        }


class SimulationResults:
    """Holds all results of a simulation."""

    moments: MomentData
    n_vs_t: Optional[ndarray] = None
    u_vs_t: Optional[ndarray] = None
    lineages: Optional[List[Lineage]] = None
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


class MultiRunStats:
    """Accumulates statistics of multiple runs."""

    nm_mean: ndarray  # Sum of all means, divide by t_tot for the actual average. Two values, one for each compartment.
    nm_sq: ndarray  # Sum of sq values. Two values, one for each compartment.
    nm_prod: int = 0  # Sum of products
    t_tot: int = 0
    n_runs_ended_early: int = 0
    n_explosions: Optional[int] = None

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]) -> "MultiRunStats":
        stats = MultiRunStats()
        stats.nm_mean = dictionary["mean"]
        stats.nm_sq = dictionary["sq"]
        stats.nm_prod = dictionary["prod"]
        stats.t_tot = dictionary["t_tot"]
        stats.n_runs_ended_early = dictionary["n_runs_ended_early"]
        stats.n_explosions = dictionary.get("n_explosions")  # This value is optional
        return stats

    def __init__(self):
        self.nm_mean = numpy.zeros(2)
        self.nm_sq = numpy.zeros(2)

    def add_results(self, results: SimulationResults):
        """Adds all relevant results of the given run to this instance."""

        # add simulated time to total time
        self.t_tot += results.run_stats.t_end
        # and accumulate statistics for each individual run
        self.nm_mean += results.moments.mean
        self.nm_sq += results.moments.sq
        self.nm_prod += results.moments.prod

        # if run ended before time (t_sim - t_tot)
        if results.run_stats.run_ended_early:
            # store this
            self.n_runs_ended_early += 1

    def to_dict(self) -> Dict[str, Any]:
        dictionary = {'mean': self.nm_mean, 'sq': self.nm_sq, 'prod': self.nm_prod, 't_tot': self.t_tot, 'n_runs_ended_early': self.n_runs_ended_early}
        if self.n_explosions is not None:
            dictionary["n_explosions"] = self.n_explosions
        return dictionary

    def print_run_statistics(self):
        nm_mean = self.nm_mean / self.t_tot
        nm_std = self.nm_sq / self.t_tot - nm_mean ** 2
        cc_NM = self.nm_prod / self.t_tot - nm_mean[0] * nm_mean[1]

        print("\t<N>=%f, s_N=%f" % (nm_mean[0], numpy.sqrt(nm_std[0])))
        print("\t<M>=%f, s_M=%f" % (nm_mean[1], numpy.sqrt(nm_std[1])))
        print("\t<N M>=%f" % cc_NM)
        print("\t<D>=%f, s_D=%f" % (nm_mean[0] + nm_mean[1], numpy.sqrt(nm_std[0] + nm_std[1] + 2 * cc_NM)))



