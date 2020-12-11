library for stem cell simulation:  
two_compartment_model_lib.py

library for lineage analysis:  
lineage_lib.py

## workflow

1. perform simulations for a parameter sweep over a range of alpha_n, alpha_m and phi with <two_compartment_model_sweep.py>. This generates an output folder <two_comp_sweep_data_fixed_D>

2. plot results with <plot_two_compartment.py>

## Workflow for space test

Use `two_compartment_model_space_test.py`

Simulate using:

`data = model.run_sim_niche( t_sim,n_max, params, n0=n0, ...)`

To change the swapping parameter `a`: `params['a'] = 0.1`
