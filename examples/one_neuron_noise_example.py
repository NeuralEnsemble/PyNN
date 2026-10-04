# encoding: utf-8
"""
A conversion to PyNN of the "One neuron with noise" example from the NEST examples gallery.

See: https://nest-simulator.readthedocs.io/en/stable/auto_examples/one_neuron_with_noise.html


Usage: one_neuron_noise_example.py [-h] [--plot-figure] [--debug DEBUG] simulator

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

sim.setup(timestep=0.01)  # (ms)

cell_type = sim.IF_curr_alpha(**cell_params)

neuron = sim.Population(1, cell_type, initial_values={"v": -70}, label="Neuron")
neuron.record(["v", "spikes"])

poisson_noise_generators = sim.Population(2, sim.SpikeSourcePoisson(rate=[80000, 15000]))
poisson_noise_generators.record("spikes")

prjs = [
    sim.Projection(
        poisson_noise_generators[0:1],
        neuron,
        sim.AllToAllConnector(),
        sim.StaticSynapse(weight=0.0012, delay=1),
        receptor_type="excitatory",
    ),
    sim.Projection(
        poisson_noise_generators[1:2],
        neuron,
        sim.AllToAllConnector(),
        sim.StaticSynapse(weight=-0.001, delay=1),
        receptor_type="inhibitory",
    ),
]

# === Run simulation ===========================================================

sim.run(1000)

data_v = neuron.get_data().segments[0].filter(name="v")[0]
data_spikes = neuron.get_data().segments[0].spiketrains

if options.plot_figure:
    figure_filename = normalized_filename(
        "Results", "one_neuron_noise_example", "png", options.simulator
    )
    Figure(
        Panel(
            data_v[:, 0],
            xlabel="Time (in ms)",
            ylabel="Membrane Potential",
            xticks=True,
            yticks=True,
        ),
        title="One neuron with noise",
        annotations=f"One neuron with noise example using PyNN (simulator: {options.simulator})",
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
