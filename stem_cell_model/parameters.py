from typing import Tuple, Dict, List, Any, Optional, Union

import numpy
from numpy.random import Generator
from numpy.random import MT19937

# The number of dividing cells used in all old simulations
# Used to load old simulation data, when this number wasn't stored
_DEFAULT_D = 30


class SimulationParameters:
    """Parameters with biophysical relevance."""
    S: int
    alpha: Tuple[float, float]
    phi: Tuple[float, float]
    T: Tuple[float, float]  # Avg, st.dev.
    n0: Tuple[int, int]  # Starting number of dividing cells per compartment.
    a: float  # Reorderings / time / cell

    @staticmethod
    def from_dict(dictionary: Dict[str, Any]):
        """Reads all parameters from a parameters dictionary."""
        if "a" not in dictionary or "n0" not in dictionary:
            dictionary = dictionary.copy()
            if "a" not in dictionary:
                # Assume well-mixed compartment - so infinite swaps
                dictionary["a"] = float("inf")
            if "n0" not in dictionary:
                # Calculate it based on S, D and alpha
                alpha = dictionary["alpha"]
                S = dictionary["S"]
                D = _DEFAULT_D
                N_0 = int(alpha[0] * S)
                M_0 = int(numpy.round(D - N_0))
                dictionary["n0"] = [N_0, M_0]

        return SimulationParameters(S=dictionary["S"],
                                    alpha=(dictionary["alpha"][0], dictionary["alpha"][1]),
                                    phi=(dictionary["phi"][0], dictionary["phi"][1]),
                                    T=(dictionary["T"][0], dictionary["T"][1]),
                                    n0=(dictionary["n0"][0], dictionary["n0"][1]),
                                    a=dictionary["a"])

    @staticmethod
    def for_D_alpha_and_phi(*, D: int, alpha_n: float, alpha_m: float, phi: float, T: Tuple[float, float],
                            a: float = float("inf")) -> Optional["SimulationParameters"]:
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
                                        n0=(int(numpy.round(N_avg)), int(numpy.round(M_avg))), a=a)
        return None

    @staticmethod
    def for_S_alpha_and_phi(*, S: int, alpha_n: float, alpha_m: float, phi: float, T: Tuple[float, float],
                            a: float = float("inf")) -> Optional["SimulationParameters"]:
        """Finds the other parameters belonging to the given ones. Returns None if the requested
         type of divisions do not exist."""

        # calculate division probabilities p_i and q_i in compartment i=n,m
        p_n = (phi + alpha_n) / 2
        q_n = (phi - alpha_n) / 2
        p_m = (phi + alpha_m) / 2
        q_m = (phi - alpha_m) / 2

        # check if division probabilities exist for this alpha_n, alpha_m and phi combination
        if (p_n >= 0) and (q_n >= 0) and (p_m >= 0) and (q_m >= 0):
            # calculate number of dividing cells D
            D = S*numpy.log(1+alpha_n)*(alpha_m-alpha_n)/alpha_m
            # calculate the average number of dividing cells in stem cell compartment
            N_avg = alpha_n * S
            if int(numpy.round(N_avg)) == 0:
                return None  # Not a simulation that will work

            # calculate the average number of dividing cells in transit amplifying compartment
            M_avg = D - N_avg

            # save parameters
            return SimulationParameters(S=int(numpy.round(S)), alpha=(alpha_n, alpha_m), phi=(phi, phi), T=T,
                                        n0=(int(numpy.round(N_avg)), int(numpy.round(M_avg))), a=a)
        return None

    def __init__(self, *, S: int, alpha: Tuple[float, float], phi: Tuple[float, float], T: Tuple[float, float], n0: Tuple[int, int], a: float = float("inf")):
        self.S = S
        self.alpha = alpha
        self.phi = phi
        self.T = T
        self.n0 = n0
        self.a = a

    @property
    def D(self) -> int:
        """The total number of dividing cells, so both in and out the niche."""
        return sum(self.n0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "S": self.S,
            "alpha": [self.alpha[0], self.alpha[1]],
            "phi": [self.phi[0], self.phi[1]],
            "T": [self.T[0], self.T[1]],
            "n0": [self.n0[0], self.n0[1]],
            "a": self.a
        }

    def __repr__(self) -> str:
        return f"SimulationParameters(S={self.S}, alpha={self.alpha}, phi={self.phi}, T={self.T}, n={self.n0}, a={self.a})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SimulationParameters):
            return False
        return other.alpha == self.alpha and other.n0 == self.n0 and other.phi == self.phi and other.T == self.T and other.a == self.a

    def __hash__(self) -> int:
        return hash((self.alpha, self.n0, self.phi, self.T, self.a))


class SimulationConfig:
    """The parameters, and all other information needed to run a simulation. Two simulations ran from exactly the same config will output exactly the same results."""
    t_sim: int  # Total simulation time
    n_max: int  # Maximum number of dividing cells. If more cells exist, the simulation fails.
    random: Generator  # Random number generator

    track_lineage_time_interval: Optional[Tuple[int, int]]  # Time points to track lineages
    track_n_vs_t: bool  # Whether the number of dividing cells over time should be stored

    def __init__(self, *, t_sim: int, n_max: int, random: Generator, track_lineage_time_interval: Optional[Tuple[int, int]] = None, track_n_vs_t: bool = False):
        self.t_sim = t_sim
        self.n_max = n_max
        self.random = random
        self.track_lineage_time_interval = track_lineage_time_interval
        self.track_n_vs_t = track_n_vs_t


def from_old_format(t_sim: int, n_max: int, params: Dict[str, Any], n0: List[int], track_lineage_time_interval: List[int], track_n_vs_t: bool) -> Tuple[SimulationConfig, SimulationParameters]:
    """Returns a SimulationConfig using the old format (with the global seed)."""
    track_lineage_time_interval_tuple = None if len(track_lineage_time_interval) != 2 else (
        track_lineage_time_interval[0], track_lineage_time_interval[1])
    params = {**params, "n0": n0}  # Don't modify original map, but add n0
    return SimulationConfig(t_sim=t_sim, n_max=n_max,
                            track_lineage_time_interval=track_lineage_time_interval_tuple,
                            track_n_vs_t=track_n_vs_t,
                            random=numpy.random.Generator(MT19937(seed=numpy.random.randint(1000000)))), \
           SimulationParameters.from_dict(params)
