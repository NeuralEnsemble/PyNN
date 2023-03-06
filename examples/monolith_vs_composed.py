"""
A demonstration of two different types of neuron model:
    - "monolithic" (pre-defined synaptic receptor types)
    - "composed" (the user can choose the synaptic receptor types)

This example shows how to build a composed model that matches the
built-in "IF_curr_exp" model, and also demonstrates one of the
additional capabilities of the composed approach, the ability to record
more of the state variables of the synapse model.

Usage: monolith_vs_composed.py [-h] [--plot-figure] [--debug DEBUG] simulator

positional arguments:
  simulator      neuron, nest, brian2 or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file.
  --debug DEBUG  Print debugging information

"""

from datetime import datetime
from pyNN.parameters import Sequence
from pyNN.utility import get_simulator, init_logging, normalized_filename


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

sim.setup(timestep=0.1, min_delay=1.0)


# === Build and instrument the network =======================================

celltype_monolith = sim.IF_curr_exp(
    tau_m=10.0, v_rest=-60.0, tau_syn_E=1.0, tau_syn_I=2.0
)

celltype_composed = sim.PointNeuron(
    sim.LIF(tau_m=10.0, v_rest=-60.0),
    excitatory=sim.CurrExpPostSynapticResponse(tau_syn=1.0),
    inhibitory=sim.CurrExpPostSynapticResponse(tau_syn=2.0),
)

neurons_monolith = sim.Population(1, celltype_monolith, initial_values={"v": -60.0})
neurons_composed = sim.Population(1, celltype_composed, initial_values={"v": -60.0})

neurons_monolith.record("v")
neurons_composed.record(["v", "excitatory.isyn", "inhibitory.isyn"])

neurons = neurons_monolith + neurons_composed

inputs = sim.Population(
    2,
    sim.SpikeSourceArray(
        spike_times=[
            Sequence([30.0]),
            Sequence([120.0]),
        ]
    ),
)

connections = {
    "excitatory": sim.Projection(
        inputs[0:1],
        neurons,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=0.5, delay=1.5),
        receptor_type="excitatory",
        label="exc",
    ),
    "inhibitory": sim.Projection(
        inputs[1:2],
        neurons,
        sim.AllToAllConnector(),
        synapse_type=sim.StaticSynapse(weight=-0.2, delay=1.5),
        receptor_type="inhibitory",
        label="inh",
    ),
}
# === Run the simulation =====================================================

sim.run(200.0)


# === Save the results, optionally plot a figure =============================

filename = "Results/monolith_vs_composed_{}.pkl".format(options.simulator)
data = neurons.get_data().segments[0]

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel

    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel(
            data.filter(name="v")[0],
            xticks=False,
            yticks=True,
            ylabel="Membrane potential (mV)",
        ),  # ylim=(-66, -48)),
        Panel(
            data.filter(name="excitatory.isyn")[0],
            xticks=False,
            yticks=True,
            ylabel="Excitatory synaptic current (nA)",
        ),
        Panel(
            data.filter(name="inhibitory.isyn")[0],
            xticks=True,
            yticks=True,
            ylabel="Inhibitory synaptic current (nA)",
            xlabel="Time (ms)"
        ),
        title="Comparing 'monolithic' and 'composed' neuron models",
        annotations=f"Simulated with {options.simulator.upper()} at {datetime.now().isoformat()}",
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
