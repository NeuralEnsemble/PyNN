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
from pyNN.utility import MultiSim, init_logging
from simple_network import SimpleNetwork

# Attributes for datastore
input = None
full_type = 'module:test_synaptic_integration'
version = '$Revision:$'


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
   
def plot_figures(spike_data, vm_data, model_parameters, interactive=False):
    pylab.rcParams['interactive'] = interactive
    if not interactive:
        pylab.rcParams['backend'] = 'Cairo'
        
    # plot Vm
    pylab.figure(1)
    for sim_name, vm in vm_data.items():
        vm['post'].plot(display=pylab.gca(), kwargs={'label': "post (%s)" % sim_name})
    pylab.legend(loc='upper left')
    ##pylab.ylim(-80, -40)
    
    # plot spikes
    pylab.figure(2)
    for i, (sim_name, spikes) in enumerate(spike_data.items()):
        if len(spikes['post']) > 0:
            pylab.plot( spikes['post'][0].spike_times, (2*i)*numpy.ones_like(spikes['post'][0].spike_times),
                       "|", label="Postsynaptic spikes (%s)" % sim_name, markersize=50)
        if len(spikes['pre']) > 0:
            print sim_name, len(spikes['pre'])
            pylab.plot( spikes['pre'][0].spike_times, (2*i+1)*numpy.ones_like(spikes['pre'][0].spike_times),
                       "|", label="Presynaptic spikes (%s)" % sim_name, markersize=50)
    pylab.plot( model_parameters.input_spike_times, (2*i+2)*numpy.ones_like(model_parameters.input_spike_times),
               "|", label="Presynaptic spikes", markersize=50 )
    pylab.ylim(-0.5,2*i+2.5)
    pylab.legend()

# ==============================================================================
if __name__ == "__main__":
    from pyNN import nest2old, neuron, pcsim
    from NeuroTools import datastore
    import pylab
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
        plot_figures(spike_data, vm_data, model_parameters)
        for fig in 1,2:
            pylab.figure(fig)
            pylab.savefig("%s/%s_%d.png" % (parameters.results_dir, ds._generate_key(this), fig))
