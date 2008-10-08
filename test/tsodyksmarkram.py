# Test of the TsodkysMarkramMechanism class

import sys
import numpy

simulator = sys.argv[1]
exec("import pyNN.%s as sim" % simulator)

sim.setup(debug=True)

spike_source = sim.Population(1, sim.SpikeSourceArray,
                              {'spike_times': numpy.arange(10, 100, 10)})
#spike_source = sim.Population(1, sim.SpikeSourcePoisson, {'rate': 50.0})

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
    populations[label].record_v()
    populations[label].record_c()
    projections[label] = sim.Projection(spike_source, populations[label], connector,
                                        target='inhibitory',
                                        synapse_dynamics=synapse_dynamics[label])
    
spike_source.record()

sim.run(200.0)

for label,p in populations.items():
    p.print_v("Results/tsodyksmarkram_%s_%s.v" % (label, simulator))
    p.print_c("Results/tsodyksmarkram_%s_%s.gsyn" % (label, simulator))
    
print spike_source.getSpikes()
    
sim.end()
