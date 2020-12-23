# encoding: utf-8
"""
Example of simple stochastic synapses

"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from pyNN.utility import get_simulator, init_logging, normalized_filename


# === Configure the simulator ================================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(quit_on_end=False)


# === Build and instrument the network =======================================

spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=np.arange(10, 100, 10)))

connector = sim.AllToAllConnector()

synapse_types = {
    'static': sim.StaticSynapse(weight=0.01, delay=0.5),
    'stochastic': sim.SimpleStochasticSynapse(p=0.5, weight=0.02, delay=0.5)
}

populations = {}
projections = {}
for label in 'static', 'stochastic':
    populations[label] = sim.Population(1, sim.IF_cond_exp(), label=label)
    populations[label].record(['v', 'gsyn_inh'])
    projections[label] = sim.Projection(spike_source, populations[label], connector,
                                        receptor_type='inhibitory',
                                        synapse_type=synapse_types[label])

spike_source.record('spikes')


# === Run the simulation =====================================================

sim.run(200.0)


# === Save the results, optionally plot a figure =============================

for label, p in populations.items():
    filename = normalized_filename("Results", "stochastic_synapses_%s" % label,
                                   "pkl", options.simulator)
    p.write_data(filename, annotations={'script_name': __file__})


if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = normalized_filename("Results", "stochastic_synapses_",
                                          "png", options.simulator)
    panels = []
    for variable in ('gsyn_inh', 'v'):
        for population in populations.values():
            panels.append(
                Panel(population.get_data().segments[0].filter(name=variable)[0],
                      data_labels=[population.label], yticks=True),
            )
    # add ylabel to top panel in each group
    panels[0].options.update(ylabel=u'Synaptic conductance (µS)')
    panels[3].options.update(ylabel='Membrane potential (mV)')
    # add xticks and xlabel to final panel
    panels[-1].options.update(xticks=True, xlabel="Time (ms)")

    Figure(*panels,
           title="Example of simple stochastic synapses",
           annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)


# === Clean up and quit ========================================================

sim.end()
