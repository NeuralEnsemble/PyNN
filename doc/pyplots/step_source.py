"""

"""

from plot_helper import plot_current_source
import pyNN.neuron as sim

sim.setup()

population = sim.Population(30, sim.IF_cond_exp(tau_m=10.0))
population[27:28].record_v()

steps = sim.StepCurrentSource(times=[50.0, 110.0, 150.0, 210.0],
                              amplitudes=[0.4, 0.6, -0.2, 0.2])
steps.inject_into(population[(6, 11, 27)])
steps._record()

sim.run(250.0)

t, i_inj = steps._get_data()
v = population.get_data().segments[0].analogsignals[0]

plot_current_source(t, i_inj, v,
                    #v_range=(-66, -49),
                    v_ticks=(-66, -64, -62, -60),
                    i_range=(-0.3, 0.7),
                    i_ticks=(-0.2, 0.0, 0.2, 0.4, 0.6),
                    t_range=(0, 250))
