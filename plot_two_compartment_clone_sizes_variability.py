"""Different clone size distributions, short term. Comparing variability."""
import numpy
from matplotlib import pyplot as plt

from stem_cell_model import clone_size_distribution_calculator
from stem_cell_model.clone_size_distribution_calculator import CloneSizeSimulationConfig
from stem_cell_model.parameters import SimulationParameters, SimulationConfig
from stem_cell_model.two_compartment_model_space import run_simulation_niche

D = 30
T = (16.153070175438597, 3.2357834505600382)  # Based on measured values

parameters_symmetric_low_noise = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.9, alpha_m=-0.9, phi=0.9, T=T, a=1/T[0])
parameters_symmetric_high_noise = SimulationParameters.for_D_alpha_and_phi(
    D=D, alpha_n=0.1, alpha_m=-0.1, phi=0.9, T=T, a=1/T[0])

random = numpy.random.Generator(numpy.random.MT19937(seed=1))
t_clone_size = 60
config = CloneSizeSimulationConfig(t_clone_size=t_clone_size, random=random, t_wait=100)

results_symmetric_low_noise = clone_size_distribution_calculator.calculate(
    run_simulation_niche, config, parameters_symmetric_low_noise)

plt.bar(results_symmetric_low_noise.indices(), results_symmetric_low_noise.to_height_array())
plt.xticks([i for i in results_symmetric_low_noise.indices() if i % 2 == 0])
plt.show()


