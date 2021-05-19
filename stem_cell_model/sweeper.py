"""Used to simulate a long list of results."""
import multiprocessing
import os
import pickle
from multiprocessing.context import Process
from typing import Callable, List, Iterable, Tuple

import numpy

from stem_cell_model.parameters import SimulationConfig, SimulationParameters
from stem_cell_model.results import SimulationResults, MultiRunStats


# Used to stop a worker process, so that it doesn't keep waiting forever on new tasks
_STOP_SIGNAL_VALUE = "~~stop~~"

# Simple type definition: a Simulator is a method that takes a SimulationConfig instance
# and outputs a SimulationResults instance
Simulator = Callable[[SimulationConfig, SimulationParameters], SimulationResults]


class _WorkPackage:
    """A package of simulations, ends up in a single file."""
    output_file: str
    params_list: List[SimulationParameters]

    def __init__(self, params_list: List[SimulationParameters], output_file: str):
        self.params_list = params_list
        self.output_file = output_file


def _worker(tasks_pending: multiprocessing.Queue, simulator: Simulator, t_sim: int, n_max: int):
    """Worker that keeps on taking tasks from a task list, and calling _sweep_single_thread on them."""
    while True:
        task = tasks_pending.get()  # This method waits until a new task becomes available
        if task == _STOP_SIGNAL_VALUE:
            break  # Finished
        worker_package: _WorkPackage = task
        _sweep_single_thread(simulator, worker_package.params_list, t_sim, n_max, worker_package.output_file)


def sweep(simulator: Simulator, params_list: List[SimulationParameters], *,
          t_sim: int, n_max: int, output_folder: str):
    """Multiprocessing parameter sweep. Uses all CPU cores so that the simulation is done as
    fast as possible. Requires the script to be ran from an "if __name__ == '__main__':"-guard."""
    worker_count = multiprocessing.cpu_count()
    os.makedirs(output_folder, exist_ok=True)

    if len(params_list) < 100:
        # Don't bother multithreading
        _sweep_single_thread(simulator, params_list, t_sim, n_max, os.path.join(output_folder, "sweep_i0.p"))
        return

    # Build a task list
    tasks_pending = multiprocessing.Queue()
    for work_package in _split_list(params_list, size_of_sublist=100, output_folder=output_folder):
        tasks_pending.put(work_package)
    # Put stoppers at the end, so that all worker processes will exit instead of waiting forever on a new task.
    # We need as many stoppers as there are worker processes so that ALL processes exit when there is no more
    # work left to be done.
    for c in range(worker_count):
        tasks_pending.put(_STOP_SIGNAL_VALUE)

    # Start all workers
    worker_processes = list()
    for i in range(worker_count):
        # Run in 8 processes
        worker_process = Process(target=_worker, args=(tasks_pending, simulator, t_sim, n_max))
        worker_process.name = f"worker-{i+1}"
        worker_processes.append(worker_process)
        worker_process.start()

    # Wait for the workers to finish
    for p in worker_processes:
        p.join()  # This method waits until the worker process is finished


def load_sweep_results(folder: str) -> Iterable[Tuple[SimulationParameters, MultiRunStats]]:
    """Loads the results of a sweep simulation."""
    for file_name in os.listdir(folder):
        if not file_name.endswith(".p"):
            continue
        with open(os.path.join(folder, file_name), "rb") as handle:
            data = pickle.load(handle)
            for params_dict, results_dict in data:
                yield SimulationParameters.from_dict(params_dict), MultiRunStats.from_dict(results_dict)


def _split_list(params_list: List[SimulationParameters], *, size_of_sublist: int,
                output_folder: str) -> Iterable[_WorkPackage]:
    """Splits the param list in work packages, each ending up in their own file."""
    sublist_start = 0
    i = 0
    while sublist_start < len(params_list):
        output_file = os.path.join(output_folder, f"sweep_i{i}.p")
        if sublist_start + size_of_sublist >= len(params_list):
            yield _WorkPackage(params_list[sublist_start:], output_file)
        else:
            yield _WorkPackage(params_list[sublist_start: sublist_start + size_of_sublist], output_file)

        sublist_start += size_of_sublist
        i += 1


def _sweep_single_thread(simulator: Simulator, params_list: List[SimulationParameters], t_sim: int, n_max: int,
                         output_file: str):
    if os.path.exists(output_file):
        return  # Nothing to do, this was done previously

    # Fixed seed (based on file name) to ensure reproducibility
    file_name = os.path.basename(output_file)

    sim_data = list()

    for i, params in enumerate(params_list):
        seed = abs(hash(params))
        random = numpy.random.Generator(numpy.random.MT19937(seed=seed))

        # print run information
        print(f"{file_name}: {i + 1}/{len(params_list)}, a_n:{params.alpha[0]:.3f}, a_m:{params.alpha[1]:.3f}, phi:{params.phi[0]:.3f}, S:{params.S}, N0:{params.n0[0]}, M0:{params.n0[1]}, seed:{seed}")

        # some simulation will end before the total simulation time t_sim because
        # stem cells are fully lost. In that case, we rerun simulations with the same 
        # initial conditions but a different random seed until the total simulation time
        # has exceeded t_sim

        # intialize statistic quantities for entire run
        output = MultiRunStats()
        
        # run simulation
        while output.t_tot < t_sim:
            config = SimulationConfig(t_sim=t_sim - output.t_tot, n_max=n_max, random=random)
            res = simulator(config, params)
            output.add_results(res)
        
        # save simulation data
        sim_data.append([params.to_dict(), output.to_dict()])
        
        # print run statistics
        output.print_run_statistics()


    # save all data in pickle file
    with open(output_file, "wb") as handle:
        pickle.dump(sim_data, handle)
