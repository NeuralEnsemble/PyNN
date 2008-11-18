"""
For each standard cell type available, creates a simple network with a single
poisson spike sources projecting to a single standard standard cell, runs the
model with each simulator, and checks that the difference in the membrane
potential trace of the post-synaptic cell between simulators is within some
tolerance.

Usage:

    python test_synaptic_integration.py PARAMETERFILE
    
Example parameter file:
{
    'sim_time': 1000.0,
    'spike_interval': 1.0,
    'seed': 876458246,
    'system': { 'timestep': 0.01,
                'min_delay': 0.1,
                'max_delay': 10.0 },
    'cell': { 'type': 'IF_curr_exp',
              'params': { 'tau_refrac': 2.0,
                          'tau_m': 20.0,
                          'tau_syn_E': 2.0 },
            },
    'weights': 0.5,
    'delays': 1.0,
    'plot_figures': True,
    'results_dir': '/home/andrew/Projects/PyNNTests/test_synaptic_integration'
}
"""

import sys
import numpy
from NeuroTools.parameters import ParameterSet
from NeuroTools.stgen import StGen
from NeuroTools.plotting import SimpleMultiplot
from pyNN.utility import MultiSim, init_logging
from simple_network import SimpleNetwork

# Attributes for datastore
input = None
full_type = 'module:test_synaptic_integration'
version = '$Revision$'


def load_parameters(url):
    return ParameterSet(url)

def run(parameters, sim_list):
    sim_time = parameters.sim_time
    spike_interval = parameters.spike_interval

    stgen = StGen()
    seed = parameters.seed
    stgen.seed(seed)

    model_parameters = ParameterSet({
        'system': parameters.system,
        'input_spike_times': stgen.poisson_generator(1000.0/spike_interval, t_stop=sim_time, array=True),
        'cell_type': parameters.cell.type,
        'cell_parameters': parameters.cell.params,
        'plasticity': { 'short_term': None, 'long_term': None },
        'weights': parameters.weights,
        'delays': parameters.delays,
    })

    networks = MultiSim(sim_list, SimpleNetwork, model_parameters)
    networks.run(sim_time)
    spike_data = networks.get_spikes()
    vm_data = networks.get_v()
    return spike_data, vm_data, model_parameters

def calc_distances(spike_data):
    distances = {'victorpurpura': {}, 'kreuz': {}}
    for measure in distances.keys():
        for sim1 in sim_list:
            distances[measure][sim1.__name__] = {} 
            for sim2 in sim_list:
                f_distance = getattr(spike_data[sim1.__name__]['post'][0], "distance_%s" % measure)
                distances[measure][sim1.__name__][sim2.__name__] = f_distance(spike_data[sim2.__name__]['post'][0])
    return distances

def plot_figure(spike_data, vm_data, model_parameters):
    fig = SimpleMultiplot(2, 1, xlabel="Time (ms)")
    panel = fig.next_panel()
    # plot Vm
    for sim_name, vm in vm_data.items():
        vm['post'].plot(display=panel, kwargs={'label': "post (%s)" % sim_name})
    panel.set_ylim(-90,-30)
    panel.legend(loc='upper right')
    # plot spikes
    panel = fig.next_panel()
    tick_labels = []
    i = 0
    for sim_name, spikes in spike_data.items():
        for pop in 'pre', 'post':
            if len(spikes[pop]) > 0:
                label = "%s (%s)" % (pop, sim_name.split('.')[1])
                panel.plot( spikes[pop][0].spike_times, i*numpy.ones_like(spikes[pop][0].spike_times),
                           "|", label=label, markersize=25)
                tick_labels.append(label)
                i += 1
    label = "Input spikes"
    panel.plot( model_parameters.input_spike_times, i*numpy.ones_like(model_parameters.input_spike_times),
               "|", label=label, markersize=25 )
    tick_labels.append(label)
    panel.set_ylim(-0.5,i+0.5)
    panel.set_yticks(range(7))
    panel.set_yticklabels(tick_labels, size=6)
    return fig

# ==============================================================================
if __name__ == "__main__":
    from pyNN import nest2old, neuron, pcsim
    from NeuroTools import datastore
    
    init_logging("test_synaptic_integration.log", debug=False)
    sim_list = [nest2old, neuron, pcsim]
    parameters = load_parameters(sys.argv[1])
    spike_data, vm_data, model_parameters = run(parameters, sim_list)
    distances = calc_distances(spike_data)
    print distances

    ds = datastore.ShelveDataStore(root_dir=parameters.results_dir, key_generator=datastore.keygenerators.hash_pickle)
    this = sys.modules[__name__]
    ds.store(this, 'distances', distances)
    ds.store(this, 'spike_data', spike_data)
    ds.store(this, 'vm_data', vm_data)
    ds.store(this, 'parameters', parameters)

    if parameters.plot_figures:
        fig = plot_figure(spike_data, vm_data, model_parameters)
        fig.save("%s/%s.png" % (parameters.results_dir, ds._generate_key(this)))