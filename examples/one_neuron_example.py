# encoding: utf-8
"""
A conversion to PyNN of the "One neuron example" from the NEST examples gallery.

See: https://nest-simulator.readthedocs.io/en/stable/auto_examples/one_neuron.html


Usage: one_neuron_example.py [-h] [--plot-figure] [--debug DEBUG] simulator

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
    "i_offset": 0.376,  # (nA)
}

# === Build the network ========================================================

sim.setup(timestep=0.1)  # (ms)

cell_type = sim.IF_curr_alpha(**cell_params)
neuron = sim.Population(1, cell_type, initial_values={"v": -70.0}, label="Simulation result")
neuron.record("v")


# === Run simulation ===========================================================

sim.run(1000.0)

data_v = neuron.get_data().segments[0].filter(name="v")[0]

if options.plot_figure:
    from neo import AsciiSignalIO
    from quantities import kHz

    reference_data = AsciiSignalIO(
        "one_neuron_example_reference_data.txt", sampling_rate=1 * kHz
    ).read_block()

    figure_filename = normalized_filename(
        "Results", "one_neuron_example", "png", options.simulator, sim.num_processes()
    )
    Figure(
        Panel(
            reference_data.segments[0].analogsignals[0],
            data_v[:, 0],
            xticks=True,
            yticks=True,
            xlabel="Time (in ms)",
            ylabel="Membrane Potential (mV)",
            line_properties=[{"lw": 3}, {"lw": 1}],
            data_labels=["Reference data", "Simulation result"]
        ),
        title="One Neuron",
        annotations=f"One Neuron Example using PyNN (simulator: {options.simulator})",
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
