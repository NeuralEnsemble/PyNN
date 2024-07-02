# encoding: utf-8
"""
A simple native NEST example of simulating One Neuron with the help of PyNN.


Usage: one_neuron_example.py [-h] [--plot-figure] [--debug DEBUG] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  plot the simulation results to a file
  --debug DEBUG  print debugging information
"""

import pyNN.nest as sim
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information."))


if options.debug:
    init_logging(None, debug=True)

# === Define parameters ========================================================

cell_params = {
    "v_rest": -70.0, # (mV)
    "v_reset": -70.0, # (mV)
    "cm": 0.250, # (nF)
    "tau_m": 10, # (ms)
    "tau_refrac": 2, # (ms)
    "tau_syn_E": 2, # (ms)
    "tau_syn_I": 2, # (ms)
    "v_thresh": -55.0, # (mV)
    "i_offset": 0.376, # (nA)
}

# === Build the network ========================================================

sim.setup(timestep=0.1) # (ms)

cell_type = sim.IF_curr_alpha(**cell_params)
neuron = sim.Population(1, cell_type, label="Neuron 1")
neuron.record("v")


# === Run simulation ===========================================================

sim.run(1000.0)

data_v = neuron.get_data().segments[0].filter(name="v")[0]

if options.plot_figure:
    figure_filename = normalized_filename("Results", "one_neuron_example", "png",
                                          options.simulator, sim.num_processes())
    Figure(
        Panel(
            data_v[:,0],
            xticks=True,
            yticks=True,
            xlabel="Time (in ms)",
            ylabel="Membrane Potential (mV)"
        ),
        title="One Neuron",
        annotations="One Neuron Example using PyNN"
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()


