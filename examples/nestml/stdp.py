"""
Example of using a cell type defined in NESTML.

This example requires that PyNN be installed using the "NESTML" option, i.e.

  $ pip install PyNN[NESTML]

or you can install NESTML directly:

  $ pip install nestml


Usage: python wang_buzsaki_synaptic_input.py [-h] [--plot-figure] simulator

positional arguments:
  simulator      nest or spinnaker

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file
  --debug        Print debugging information

"""

import os
from pyNN.utility import get_simulator, init_logging, normalized_filename, SimulationProgressBar
from pyNN.random import NumpyRNG, RandomDistribution


# === Configure the simulator ================================================

sim, options = get_simulator(
    (
        "--plot-figure",
        "Plot the simulation results to a file.",
        {"action": "store_true"},
    ),
    ("--debug", "Print debugging information"),
)

if options.debug:
    init_logging(None, debug=True)

# === Register the NESTML synapse (and co-generated neuron) before sim.setup()

current_dir = os.path.dirname(os.path.abspath(__file__))
synapse_type = sim.nestml.nestml_synapse_type(
    "stdp_synapse",
    os.path.join(current_dir, "stdp_synapse.nestml"),
    postsynaptic_neuron_nestml_description=os.path.join(current_dir, "iaf_psc_exp_neuron.nestml"),
)
post_cell_type = synapse_type.postsynaptic_cell_type

sim.setup(timestep=0.01, min_delay=1.0)


# === Build and instrument the network =======================================

p2 = sim.Population(10, sim.SpikeSourcePoisson())
p1 = sim.Population(10, post_cell_type())

connector = sim.AllToAllConnector()

prj = sim.Projection(p2, p1, connector, receptor_type='excitatory', synapse_type=synapse_type())

p1.record(["V_m", "I_syn_exc", "I_syn_inh"])

# === Run the simulation =====================================================

print("Running simulation")
t_stop = 100.0
pb = SimulationProgressBar(t_stop / 10, t_stop)

sim.run(t_stop, callbacks=[pb])


# === Save the results, optionally plot a figure =============================

print("Saving results")
filename = normalized_filename("Results", "nestml_example", "pkl", options.simulator)
p1.write_data(filename, annotations={"script_name": __file__})

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel

    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel(
            p1.get_data().segments[0].filter(name="V_m")[0],
            ylabel="Membrane potential (mV)",
            data_labels=[p1.label],
            yticks=True,
        ),
        Panel(
            p1.get_data().segments[0].filter(name="I_syn_exc")[0],
            ylabel="Excitatory synaptic current (pA)",
            data_labels=[p1.label],
            yticks=True,
            #ylim=(0, 1),
        ),
        Panel(
            p1.get_data().segments[0].filter(name="I_syn_inh")[0],
            xticks=True,
            xlabel="Time (ms)",
            ylabel="Inhibitory synaptic current (pA)",
            data_labels=[p1.label],
            yticks=True,
            #ylim=(0, 1),
        ),
        title="Responses of Wang-Buzsaki neuron model, defined in NESTML, to synaptic input",
        annotations="Simulated with %s" % options.simulator.upper(),
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
