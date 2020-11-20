library for stem cell simulation:  
two_compartment_model_lib.py

library for lineage analysis:  
lineage_lib.py

workflow:

1. perform simulations for a parameter sweep over a range of alpha_n, alpha_m and phi with <two_compartment_model_sweep.py>. This generates an output folder <two_comp_sweep_data_fixed_D>

2. plot results with <plot_two_compartment.py>

