"""
Example of depressing and facilitating synapses

Andrew Davison, UNIC, CNRS
May 2009

$Id$
"""

import numpy
from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  
exec("import pyNN.%s as sim" % simulator_name)

sim.setup(quit_on_end=False)

spike_source = sim.Population(1, sim.SpikeSourceArray,
                              {'spike_times': numpy.arange(10, 100, 10)})

connector = sim.AllToAllConnector(weights=0.01, delays=0.5)

synapse_dynamics = {
    'static': None,
    'depressing': sim.SynapseDynamics(
        fast=sim.TsodyksMarkramMechanism(U=0.5, tau_rec=800.0, tau_facil=0.0)),
    'facilitating': sim.SynapseDynamics(
        fast=sim.TsodyksMarkramMechanism(U=0.04, tau_rec=100.0, tau_facil=1000.0)),
}

populations = {}
projections = {}
for label in 'static', 'depressing', 'facilitating':
    populations[label] = sim.Population(1, sim.IF_cond_exp, {'e_rev_I': -75}, label=label)
    populations[label].record('v')
    if populations[label].can_record('gsyn'):
        populations[label].record(['gsyn_exc', 'gsyn_inh'])
    projections[label] = sim.Projection(spike_source, populations[label], connector,
                                        target='inhibitory',
                                        synapse_dynamics=synapse_dynamics[label])
    
spike_source.record('spikes')

sim.run(200.0)

for label,p in populations.items():
    p.write_data("Results/tsodyksmarkram_%s_%s.txt" % (label, simulator_name))
    
print spike_source.get_data('spikes')
    
sim.end()
