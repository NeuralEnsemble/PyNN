"""
A demonstration of the responses of different standard neuron models to current injection.

Usage: python cell_type_demonstration.py [-h] [--plot-figure] [--debug] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file.
  --debug        Print debugging information

"""

from pyNN.utility import get_simulator, init_logging, normalized_filename


# === Configure the simulator ================================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(timestep=0.01, min_delay=1.0)


# === Build and instrument the network =======================================

cuba_exp = sim.Population(1, sim.IF_curr_exp(i_offset=1.0), label="IF_curr_exp")
hh = sim.Population(1, sim.HH_cond_exp(i_offset=0.2), label="HH_cond_exp")
adexp = sim.Population(1, sim.EIF_cond_exp_isfa_ista(i_offset=1.0), label="EIF_cond_exp_isfa_ista")
adapt = sim.Population(1, sim.IF_cond_exp_gsfa_grr(i_offset=2.0), label="IF_cond_exp_gsfa_grr")
izh = sim.Population(1, sim.Izhikevich(i_offset=0.01), label="Izhikevich")

all_neurons = cuba_exp + hh + adexp + adapt + izh

all_neurons.record('v')
adexp.record('w')
izh.record('u')


# === Run the simulation =====================================================

sim.run(100.0)


# === Save the results, optionally plot a figure =============================

filename = normalized_filename("Results", "cell_type_demonstration", "pkl", options.simulator)
all_neurons.write_data(filename, annotations={'script_name': __file__})

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel(cuba_exp.get_data().segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[cuba_exp.label], yticks=True, ylim=(-66, -48)),
        Panel(hh.get_data().segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[hh.label], yticks=True, ylim=(-100, 60)),
        Panel(adexp.get_data().segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[adexp.label], yticks=True, ylim=(-75, -40)),
        Panel(adexp.get_data().segments[0].filter(name='w')[0],
              ylabel="w (nA)",
              data_labels=[adexp.label], yticks=True, ylim=(0, 0.4)),
        Panel(adapt.get_data().segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[adapt.label], yticks=True, ylim=(-75, -45)),
        Panel(izh.get_data().segments[0].filter(name='v')[0],
              ylabel="Membrane potential (mV)",
              data_labels=[izh.label], yticks=True, ylim=(-80, 40)),
        Panel(izh.get_data().segments[0].filter(name='u')[0],
              xticks=True, xlabel="Time (ms)",
              ylabel="u (mV/ms)",
              data_labels=[izh.label], yticks=True, ylim=(-14, 0)),
        title="Responses of standard neuron models to current injection",
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
