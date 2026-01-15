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

sim.setup(timestep=0.01, min_delay=1.0)

import pyNN
nest = pyNN.nest
from pyNN.utility import init_logging

# === Create the cell type from a NESTML definition

current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, "wb_cond_exp_neuron.nestml")
# celltype_cls = sim.nestml.nestml_cell_type("wb_cond_exp_neuron", input_path)

synapse_type, post_cell_type = sim.nestml.nestml_synapse_type("stdp_synapse", "stdp_synapse.nestml", "iaf_psc_exp_neuron.nestml")


# === Build and instrument the network =======================================

init_logging(logfile=None, debug=True)

# nest.setup()

p2 = nest.Population(10, nest.SpikeSourcePoisson())
p1 = nest.Population(10, post_cell_type())

connector = nest.AllToAllConnector()

# stdp_params = {'Wmax': 50.0, 'lambda': 0.015, 'w': 0.001}
stdp_params = {}
prj = nest.Projection(p2, p1, connector, receptor_type='excitatory', synapse_type=synapse_type(**stdp_params))

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
