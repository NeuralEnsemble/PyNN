"""

"""

from plot_helper import plot_current_source
import pyNN.neuron as sim

sim.setup()

population = sim.Population(10, sim.IF_cond_exp(tau_m=10.0))
population[3:4].record_v()

pulse = sim.DCSource(amplitude=0.5, start=20.0, stop=80.0)
pulse.inject_into(population[3:7])
pulse._record()

sim.run(100.0)

t, i_inj = pulse._get_data()
v = population.get_data().segments[0].analogsignals[0]

plot_current_source(t, i_inj, v,
                    v_range=(-65.5, -59.5),
                    v_ticks=(-65, -64, -63, -62, -61, -60),
                    i_range=(-0.1, 0.55),
                    i_ticks=(0.0, 0.2, 0.4))
