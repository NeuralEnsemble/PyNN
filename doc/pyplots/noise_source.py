"""

"""

from plot_helper import plot_current_source
import pyNN.neuron as sim

sim.setup()

population = sim.Population(30, sim.IF_cond_exp(tau_m=10.0))
population[0:1].record_v()

noise = sim.NoisyCurrentSource(mean=1.5, stdev=1.0, start=50.0, stop=450.0,
                               dt=1.0)
population.inject(noise)
noise._record()

sim.run(500.0)

t, i_inj = noise._get_data()
v = population.get_data().segments[0].analogsignals[0]

plot_current_source(t, i_inj, v,
                    v_range=(-66, -48),
                    v_ticks=(-65, -60, -55, -50),
                    i_range=(-3, 5),
                    i_ticks=range(-2, 6, 2),
                    t_range=(0, 500))
