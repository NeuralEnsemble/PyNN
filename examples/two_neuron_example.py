"""
A conversion to PyNN of the "Two neuron example" from the NEST examples gallery.

See: https://nest-simulator.readthedocs.io/en/stable/auto_examples/twoneurons.html

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
    "v_rest": -70.0,  # (mV)
    "v_reset": -70.0,  # (mV)
    "cm": 0.250,  # (nF)
    "tau_m": 10,  # (ms)
    "tau_refrac": 2,  # (ms)
    "tau_syn_E": 2,  # (ms)
    "tau_syn_I": 2,  # (ms)
    "v_thresh": -55.0,  # (mV)
    "i_offset": 0,  # (nA)
}

# === Build the network ========================================================

sim.setup(timestep=0.1)  # (ms)

cell_type = sim.IF_curr_alpha(**cell_params)
neurons = sim.Population(2, cell_type, initial_values={"v": -70}, label="Neurons")
neurons[0:1].set(i_offset=0.376)
neurons.record("v")

syn = sim.StaticSynapse(weight=0.02, delay=1.0)
projection1 = sim.Projection(neurons[0:1], neurons[1:2], sim.AllToAllConnector(), syn)

# === Run simulation ===========================================================

sim.run(1000)

data = neurons.get_data().segments[0].filter(name="v")[0]

if options.plot_figure:
    figure_filename = normalized_filename(
        "Results", "two_neuron_example", "png", options.simulator
    )

    Figure(
        Panel(
            data,
            xticks=True,
            yticks=True,
            xlabel="Time (in ms)",
            ylabel="Membrane Potential (in mV)",
        ),
        title="Two neurons",
        annotations=f"Two neuron example using PyNN (simulator: {options.simulator})",
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
