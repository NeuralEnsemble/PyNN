"""
Two Neuron Example translations from NEST into PyNN.

Usage: two_neuron_example.py [-h] [--plot-figure] [--debug DEBUG] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  plot the simulation results to a file
  --debug DEBUG  print debugging information
"""

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
    "i_offset": 0, # (nA)
}

# === Build the network ========================================================

sim.setup(timestep=0.1) # (ms)

cell_type = sim.IF_curr_alpha(**cell_params)
neuron1 = sim.Population(1, cell_type, label="Neuron 1")
neuron1.set(i_offset=0.376)
neuron1.record("v")

neuron2 = sim.Population(1, cell_type, label="Neuron 2")
neuron2.record("v")

syn = sim.StaticSynapse(weight=0.02, delay=1.0)
projection1 = sim.Projection(neuron1, neuron2, sim.AllToAllConnector(), syn)

# === Run simulation ===========================================================

sim.run(1000)

neuron1_data_v = neuron1.get_data().segments[0].filter(name="v")[0]
neuron2_data_v = neuron2.get_data().segments[0].filter(name="v")[0]

if options.plot_figure:
    figure_filename = normalized_filename("Results", "two_neuron_example", "png",
                                          options.simulator, sim.num_processes())

    Figure(
        Panel(neuron1_data_v[:,0],
            xticks=True,
            yticks=True,
            xlabel="Time (in ms)",
            ylabel="Membrane Potential (in mV)"),
        Panel(neuron2_data_v[:,0],
            xticks=True,
            yticks=True,
            xlabel="Time (in ms)",
            ylabel="Membrane Potential (in mV)")
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()