"""
Example of using PyNN with Neo output and the reset() function
"""

from pyNN.utility import get_script_args
#import pyNN.neuron as sim  # can of course replace `neuron` with `nest`, `brian`, etc.
simulator_name = get_script_args(1)[0]  
exec("import pyNN.%s as sim" % simulator_name)
import matplotlib.pyplot as plt
from quantities import nA

sim.setup()

cell = sim.Population(1, sim.HH_cond_exp)
step_current = sim.DCSource(start=20.0, stop=80.0)
step_current.inject_into(cell)

cell.record('v')

for amp in (-0.2, -0.1, 0.0, 0.1, 0.2):
    step_current.amplitude = amp
    sim.run(100.0)
    sim.reset(annotations={"amplitude": amp*nA})
    
data = cell.get_data()

sim.end()

for segment in data.segments:
    vm = segment.analogsignalarrays[0]
    plt.plot(vm.times, vm,
             label=str(segment.annotations["amplitude"]))
plt.legend(loc="upper left")
plt.xlabel("Time (%s)" % vm.times.units._dimensionality)
plt.ylabel("Membrane potential (%s)" % vm.units._dimensionality)
plt.savefig("reset_example_%s.png" % simulator_name)