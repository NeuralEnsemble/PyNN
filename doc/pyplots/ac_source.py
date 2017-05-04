"""

"""

from plot_helper import plot_current_source
import pyNN.neuron as sim

sim.setup()

population = sim.Population(10, sim.IF_cond_exp(tau_m=10.0))
population[0:1].record_v()

sine = sim.ACSource(start=50.0, stop=450.0, amplitude=1.0, offset=1.0,
                    frequency=10.0, phase=180.0)
population.inject(sine)
sine._record()

sim.run(500.0)

t, i_inj = sine._get_data()
v = population.get_data().segments[0].analogsignals[0]

plot_current_source(t, i_inj, v,
                    v_range=(-66, -49),
                    v_ticks=(-65, -60, -55, -50),
                    i_range=(-0.1, 2.1),
                    i_ticks=(0.0, 0.5, 1.0, 1.5),
                    t_range=(0, 500))
