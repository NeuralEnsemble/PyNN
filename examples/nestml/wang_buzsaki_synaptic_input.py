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

# === Register the NESTML cell type (must happen before sim.setup())

current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, "wb_cond_exp_neuron.nestml")
celltype_cls = sim.nestml.nestml_cell_type("wb_cond_exp_neuron", input_path)

sim.setup(timestep=0.01, min_delay=1.0)

# add some variability between neurons
rng = NumpyRNG(seed=1309463846)
rnd = lambda min, max: RandomDistribution("uniform", (min, max), rng=rng)

celltype = celltype_cls(
    g_Na=rnd(3400, 3600),  # nS
    g_K=rnd(850, 950),  # nS
    g_L=rnd(8, 12),  # nS
    C_m=rnd(80, 120),  # pF
    V_Tr=rnd(-56, -54),  # mV
    I_e=0,  # pA
)


# === Build and instrument the network =======================================

inputs = sim.Population(5, sim.SpikeSourcePoisson(rate=100.0), label="input spikes")
cells = sim.Population(5, celltype, label=celltype_cls.__name__)
cells.record(["V_m", "I_syn_exc", "I_syn_inh"])

print("Connecting cells")
syn = sim.StaticSynapse(weight=0.01, delay=1.0)
prj = sim.Projection(inputs, cells, sim.FixedNumberPostConnector(3), syn, receptor_type="excitatory")

# === Run the simulation =====================================================

print("Running simulation")
t_stop = 100.0
pb = SimulationProgressBar(t_stop / 10, t_stop)

sim.run(t_stop, callbacks=[pb])


# === Save the results, optionally plot a figure =============================

print("Saving results")
filename = normalized_filename("Results", "nestml_example", "pkl", options.simulator)
cells.write_data(filename, annotations={"script_name": __file__})

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel

    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel(
            cells.get_data().segments[0].filter(name="V_m")[0],
            ylabel="Membrane potential (mV)",
            data_labels=[cells.label],
            yticks=True,
        ),
        Panel(
            cells.get_data().segments[0].filter(name="I_syn_exc")[0],
            ylabel="Excitatory synaptic current (pA)",
            data_labels=[cells.label],
            yticks=True,
            #ylim=(0, 1),
        ),
        Panel(
            cells.get_data().segments[0].filter(name="I_syn_inh")[0],
            xticks=True,
            xlabel="Time (ms)",
            ylabel="Inhibitory synaptic current (pA)",
            data_labels=[cells.label],
            yticks=True,
            #ylim=(0, 1),
        ),
        title="Responses of Wang-Buzsaki neuron model, defined in NESTML, to synaptic input",
        annotations="Simulated with %s" % options.simulator.upper(),
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
