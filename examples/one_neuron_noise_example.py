import pyNN.nest as sim
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

sim.setup(timestep=0.1) # (ms)


cell_params = {
    "v_rest": -70.0, # (mV)
    "v_reset": -70.0, # (mV)
    "cm": 0.250, # (nF)
    "tau_m": 10, # (ms)
    "tau_refrac": 2, # (ms)
    "tau_syn_E": 2, # (ms)
    "tau_syn_I": 2, # (ms)
    "v_thresh": -55.0, # (mV)
    "i_offset": 0, # (nA)
}


cell_type = sim.IF_curr_alpha(**cell_params)

neuron = sim.Population(1, cell_type, label="Neuron")
neuron.record(["v", "spikes"])

poisson_noise_generators = sim.Population(2, sim.SpikeSourcePoisson(rate=[80000, 15000]))
poisson_noise_generators.record("spikes")

syn = sim.StaticSynapse(weight=0.0012, delay=1)

prj = sim.Projection(poisson_noise_generators, neuron, sim.AllToAllConnector() , syn)
prj.setWeights([0.0012, 0.001])
prj.setDelays([1, 1])

sim.run(1000)

data_v = neuron.get_data().segments[0].filter(name="v")[0]
data_spikes = neuron.get_data().segments[0].spiketrains

Figure(
    Panel(data_v[:,0], xlabel="Time (in ms)", ylabel="Membrane Potential", xticks=True, yticks=True)
).show()

