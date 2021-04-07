from typing import Tuple, Dict, List, Any, Optional, Union

import numpy
from numpy.random import Generator
from numpy.random import MT19937


class SimulationParameters:
    """Parameters with biophysical relevance."""
    S: int
    alpha: Tuple[float, float]
    phi: Tuple[float, float]
    T: Tuple[float, float]  # Avg, st.dev.
    n0: Tuple[int, int]  # Starting number of dividing cells per compartment.
    a: float  # Reorderings / time / cell


    @staticmethod
    def from_old_format(params: Dict[str, Any], n0: List[int]) -> "SimulationParameters":
        """For compatibility with how the method used to be called."""
        return SimulationParameters(S=params["S"],
                                    alpha=(params["alpha"][0], params["alpha"][1]),
                                    phi=(params["phi"][0], params["phi"][1]),
                                    T=(params["T"][0], params["T"][1]),
                                    n=(n0[0], n0[1]))

    @staticmethod
    def for_d_alpha_and_phi(D: int, alpha_n: float, alpha_m: float, phi: float) -> Optional["SimulationParameters"]:
        """Finds the other parameters belonging to the given ones. Returns None if the requested
         type of divisions do not exist."""

        # calculate division probabilities p_i and q_i in compartment i=n,m
        p_n = (phi + alpha_n) / 2
        q_n = (phi - alpha_n) / 2
        p_m = (phi + alpha_m) / 2
        q_m = (phi - alpha_m) / 2

        # check if division probabilities exist for this alpha_n, alpha_m and phi combination
        if (p_n >= 0) and (q_n >= 0) and (p_m >= 0) and (q_m >= 0):
            # calculate compartment size S
            S = D / numpy.log(1 + alpha_n) * alpha_m / (alpha_m - alpha_n)
            # calculate the average number of dividing cells in stem cell compartment
            N_avg = alpha_n * S
            # calculate the average number of dividing cells in transit amplifying compartment
            M_avg = D - N_avg

            # save parameters
            return SimulationParameters(S=int(numpy.round(S)), alpha=(alpha_n, alpha_m), phi=(phi, phi), T=T,
                                        n=(int(numpy.round(N_avg)), int(numpy.round(D - N_avg))))
        return None

    def __init__(self, *, S: int, alpha: Tuple[float, float], phi: Tuple[float, float], T: Tuple[float, float], n: Tuple[int, int], a: float = 0):
        self.S = S
        self.alpha = alpha
        self.phi = phi
        self.T = T
        self.n0 = n
        self.a = a


class SimulationConfig:
    """The parameters, and all other information needed to run a simulation. Two simulations ran from exactly the same config will output exactly the same results."""
    t_sim: int  # Total simulation time
    n_max: int  # Maximum number of dividing cells. If more cells exist, the simulation fails.
    random: Generator  # Random number generator

    params: SimulationParameters  # Biophysical parameters.
    track_lineage_time_interval: Optional[Tuple[int, int]]  # Time points to track lineages
    track_n_vs_t: bool  # Whether the number of dividing cells over time should be stored

    @staticmethod
    def from_old_format(t_sim: int, n_max: int, params: Dict[str, Any], n0: List[int], track_lineage_time_interval: List[int], track_n_vs_t: bool):
        """Returns a SimulationConfig using the old format (with the global seed)."""
        track_lineage_time_interval_tuple = None if len(track_lineage_time_interval) != 2 else (
            track_lineage_time_interval[0], track_lineage_time_interval[1])
        return SimulationConfig(t_sim=t_sim, n_max=n_max,
                                params=SimulationParameters.from_old_format(params, n0),
                                track_lineage_time_interval=track_lineage_time_interval_tuple,
                                track_n_vs_t=track_n_vs_t,
                                random=numpy.random.Generator(MT19937(seed=numpy.random.randint(1000000))))

    def __init__(self, *, t_sim: int, n_max: int, random: Generator, params: SimulationParameters, track_lineage_time_interval: Optional[Tuple[int, int]] = None, track_n_vs_t: bool = False):
        self.t_sim = t_sim
        self.n_max = n_max
        self.random = random
        self.params = params
        self.track_lineage_time_interval = track_lineage_time_interval
        self.track_n_vs_t = track_n_vs_t
