# encoding: utf-8
"""
...

"""

from pyNN.utility import get_simulator, init_logging, normalized_filename
import neo
import numpy as np
import matplotlib
matplotlib.use('Agg')


# === Configure the simulator ================================================

sim, options = get_simulator(("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
                             ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(quit_on_end=False)


# === Build and instrument the network =======================================

spike_times = np.hstack((np.arange(10, 100, 10), np.arange(250, 350, 10)))
spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

connector = sim.AllToAllConnector()

depressing = dict(U=0.8, tau_rec=100.0, tau_facil=0.0, weight=0.01, delay=0.5)
facilitating = dict(U=0.04, tau_rec=50.0, tau_facil=200.0, weight=0.01, delay=0.5)

synapse_types = {
    'depressing, n=1':   sim.MultiQuantalSynapse(n=1, **depressing),
    'depressing, n=5':   sim.MultiQuantalSynapse(n=5, **depressing),
    'facilitating, n=1': sim.MultiQuantalSynapse(n=1, **facilitating),
    'facilitating, n=5': sim.MultiQuantalSynapse(n=5, **facilitating),
}

populations = {}
projections = {}
for label in synapse_types:
    populations[label] = sim.Population(
        1000, sim.IF_cond_exp(e_rev_I=-75, tau_syn_I=4.3), label=label)
    populations[label].record('gsyn_inh')
    projections[label] = sim.Projection(spike_source, populations[label], connector,
                                        receptor_type='inhibitory',
                                        synapse_type=synapse_types[label])
    projections[label].initialize(a=synapse_types[label].parameter_space['n'],
                                  u=synapse_types[label].parameter_space['U'])

spike_source.record('spikes')

#if "nest" in sim.__name__:
#    print(sim.nest.GetStatus([projections['depressing, n=5'].nest_connections[0]]))

# === Run the simulation =====================================================

sim.run(400.0)


# === Save the results, optionally plot a figure =============================

for label, p in populations.items():
    filename = normalized_filename("Results", "multiquantal_synapses_%s" % label,
                                   "pkl", options.simulator)
    p.write_data(filename, annotations={'script_name': __file__})


if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    # figure_filename = normalized_filename("Results", "multiquantal_synapses",
    #                                      "png", options.simulator)
    figure_filename = "Results/multiquantal_synapses_{}.png".format(options.simulator)

    data = {}
    for label in synapse_types:
        data[label] = populations[label].get_data().segments[0]
        gsyn = data[label].filter(name='gsyn_inh')[0]
        gsyn_mean = neo.AnalogSignal(gsyn.mean(axis=1).reshape(-1, 1),
                                     sampling_rate=gsyn.sampling_rate,
                                     array_annotations={"channel_index": np.array([0])})
        gsyn_mean.name = 'gsyn_inh_mean'
        data[label].analogsignals.append(gsyn_mean)

    def make_panel(population, label):
        return Panel(population.get_data().segments[0].filter(name='gsyn_inh')[0],
                     data_labels=[label], yticks=True)
    labels = sorted(synapse_types)
    panels = [
        Panel(data[label].filter(name='gsyn_inh_mean')[0],
              data_labels=[label], yticks=True)
        for label in labels
    ]
    # add ylabel to top panel in each group
    panels[0].options.update(ylabel=u'Synaptic conductance (µS)')
    ##panels[len(synapse_types)].options.update(ylabel='Membrane potential (mV)')
    # add xticks and xlabel to final panel
    panels[-1].options.update(xticks=True, xlabel="Time (ms)")

    Figure(*panels,
           title="Example of facilitating and depressing multi-quantal release synapses",
           annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)


# === Clean up and quit ========================================================

sim.end()
