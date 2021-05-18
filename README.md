library for stem cell simulation:  
two_compartment_model_lib.py

library for lineage analysis:  
lineage_lib.py

## Workflow for reproducing results

1. perform simulations for a parameter sweep over a range of alpha_n, alpha_m and phi with <two_compartment_model_sweep.py>. This generates an output folder <two_comp_sweep_data_fixed_D>

2. plot results with <plot_two_compartment.py>

## Running your own simulation for certain parameters

```python
import numpy

from stem_cell_model import tools
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.results import MultiRunStats
from stem_cell_model.two_compartment_model_space import run_simulation_niche

T = (16.153070175438597, 3.2357834505600382)
params = SimulationParameters.for_S_alpha_and_phi(S=1114, alpha_n=0.025, alpha_m=-0.275, phi=1.0, T=T, a=100/T[0])
random = numpy.random.Generator(numpy.random.MT19937(seed=1))
total_run_time = 100000

# Run the actual simulation
run_data = MultiRunStats()
while run_data.t_tot < total_run_time:
    config = SimulationConfig(t_sim=total_run_time - run_data.t_tot, n_max=5*30, random=random)
    results = run_simulation_niche(config, params)
    run_data.add_results(results)

# Collect results
stats = tools.get_single_parameter_set_statistics(run_data)
```
The stats object has properties like `stats.n_mean`. You can store simulation results using Pickle and `run_data.to_dict()`.

## Running a parameter sweep
See for example [two_compartment_model_sweep_well_mixed.py](./two_compartment_model_sweep_well_mixed.py) for a concise example. This scripts saves the results to a folder. It uses the `stem_cell_model.sweeper` module, which uses all cores of your PC.

Read back the results using `stem_cell_model.sweeper.load_sweep_results(folder)`.
