import pyNN.neuron as sim  # can of course replace `nest` with `neuron`, `brian`, etc.
import matplotlib.pyplot as plt
from quantities import nA

sim.setup()

cell = sim.Population(1, sim.HH_cond_exp())
step_current = sim.DCSource(start=20.0, stop=80.0)
step_current.inject_into(cell)

cell.record('v')

for amp in (-0.2, -0.1, 0.0, 0.1, 0.2):
    step_current.amplitude = amp
    sim.run(100.0)
    sim.reset(annotations={"amplitude": amp * nA})

data = cell.get_data()

sim.end()

for segment in data.segments:
    vm = segment.analogsignals[0]
    plt.plot(vm.times, vm,
             label=str(segment.annotations["amplitude"]))
plt.legend(loc="upper left")
plt.xlabel("Time (%s)" % vm.times.units._dimensionality)
plt.ylabel("Membrane potential (%s)" % vm.units._dimensionality)

plt.show()
