"""

"""

import matplotlib.pyplot as plt
import pyNN.neuron as sim

sim.setup()

population = sim.Population(10, sim.IF_cond_exp, {'tau_m': 10.0})

pulse = sim.DCSource(dict(amplitude=0.5, start=20.0, stop=80.0))
pulse.inject_into(population[3:7])

population[3:4].record_v()

sim.run(100.0)

id, t, v = population.get_v().T

plt.figure(figsize=(12, 3))

plt.plot(t, v)
plt.ylim(-66, -59)
plt.xlabel('Time (ms)') # this is not shown with a 12x3 figsize
plt.ylabel('V (mV)')

plt.show()
