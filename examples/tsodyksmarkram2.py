"""
Example of depressing and facilitating, current-based, alpha synapses

Andrew Davison, UNIC, CNRS
May 2009

$Id:$
"""

import numpy
from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]
exec("import pyNN.%s as sim" % simulator_name)

sim.setup()

spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=numpy.arange(10, 100, 10)))

connector = sim.AllToAllConnector()

synapse_types = {
    'static': sim.StaticSynapse(weight=-1.0, delay=0.5),
    'depressing': sim.TsodyksMarkramSynapse(U=0.5, tau_rec=800.0, tau_facil=0.0,
                                            weight=-1.0, delay=0.5),
    'facilitating': sim.TsodyksMarkramSynapse(U=0.04, tau_rec=100.0, tau_facil=1000.0,
                                            weight=-1.0, delay=0.5),
}

populations = {}
projections = {}
for label in 'static', 'depressing', 'facilitating':
    populations[label] = sim.Population(1, sim.IF_curr_alpha(), label=label)
    populations[label].record_v()
    projections[label] = sim.Projection(spike_source, populations[label], connector,
                                        receptor_type='inhibitory',
                                        synapse_type=synapse_types[label])

spike_source.record('spikes')

sim.run(200.0)

for label,p in populations.items():
    p.write_data("Results/tsodyksmarkram2_%s_%s.pkl" % (label, simulator_name))

sim.end()
