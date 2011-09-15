"""
Network of integrate-and-fire neurons with distance-dependent connectivity and STDP.
"""

from pyNN.utility import get_script_args
usage = """Usage: python stdp_network.py <simulator>"""
simulator_name, = get_script_args(1, usage)  
exec("import pyNN.%s as sim" % simulator_name)
from pyNN import space

n_exc = 80  
n_inh = 20
n_stim = 20
cell_parameters = {
    'tau_m' : 20.0,    'tau_syn_E': 2.0,    'tau_syn_I': 5.0,
    'v_rest': -65.0,   'v_reset'  : -70.0,  'v_thresh':  -50.0,
    'cm':     1.0,     'tau_refrac': 2.0,   'e_rev_E':   0.0,
    'e_rev_I': -70.0,
}
grid_parameters = {
    'aspect_ratio': 1, 'dx': 50.0, 'dy': 50.0, 'fill_order': 'random'
}
stimulation_parameters = {
    'rate': 100.0,
    'duration': 50.0
}
stdp_parameters = {
    'timing_dependence': {'tau_plus': 20.0, 'tau_minus': 20.0},
    'weight_dependence': {'w_min':0, 'w_max': 0.04, 'A_plus': 0.01, 'A_minus': 0.012}    
}
connectivity_parameters = {
    'gaussian': {'d_expression': 'exp(-d**2/1e4)', 'weights': 0.01, 'delays': '0.1+0.001*d'},
    'global': {'p_connect': 0.1, 'weights': 0.05, 'delays': '0.1+0.001*d'},
    'input': {'n': 10, 'weights': 0.01, 'delays': 0.1},
}

sim.setup()

all_cells = sim.Population(n_exc+n_inh, sim.IF_cond_exp, cell_parameters,
                           structure=space.Grid2D(**grid_parameters),
                           label="All Cells")
exc_cells = all_cells[:n_exc]; exc_cells.label = "Excitatory cells"
inh_cells = all_cells[n_exc:]; inh_cells.label = "Inhibitory cells"

ext_stim = sim.Population(n_stim, sim.SpikeSourcePoisson, stimulation_parameters,
                          label="External Poisson stimulation")

stdp_mechanism = sim.STDPMechanism(
                    timing_dependence=sim.SpikePairRule(**stdp_parameters['timing_dependence']),
                    weight_dependence=sim.AdditiveWeightDependence(**stdp_parameters['weight_dependence']))

gaussian_connectivity = sim.DistanceDependentProbabilityConnector(
                            **connectivity_parameters['gaussian'])
global_connectivity = sim.FixedProbabilityConnector(
                            **connectivity_parameters['global'])
input_connectivity = sim.FixedNumberPostConnector(
                            **connectivity_parameters['input'])

exc_connections = sim.Projection(exc_cells, all_cells,
                                 gaussian_connectivity,
                                 target='excitatory',
                                 synapse_dynamics=sim.SynapseDynamics(slow=stdp_mechanism),
                                 label='Excitatory connections')

inh_connections = sim.Projection(inh_cells, all_cells,
                                 global_connectivity,
                                 target='inhibitory',
                                 label='Inhibitory connections')

stim_connections = sim.Projection(ext_stim, all_cells,
                                  input_connectivity,
                                  target='excitatory',
                                  label='Input connections')

print __doc__
print "The network consists of:\n"
print all_cells.describe()
print exc_cells.describe()
print inh_cells.describe()
print ext_stim.describe()
print "connected as follows:\n"
print exc_connections.describe()
print inh_connections.describe()
print stim_connections.describe()

sim.end()