"""
Simple test of recording a SpikeSourcePoisson object

Andrew Davison, UNIC, CNRS
September 2006

$Id$
"""

from pyNN.utility import get_script_args, init_logging

simulator_name = get_script_args(1)[0]
exec("from pyNN.%s import *" % simulator_name)

init_logging(None, debug=True)

setup(timestep=0.01, min_delay=0.1)

poissonsource = Population(10, SpikeSourcePoisson,
                           {'rate': 100.0, 'duration': 100.0, 'start': 100.0})
poissonsource.record('spikes')

run(300.0)

print "Mean spike count:", poissonsource.mean_spike_count()
print "First few spikes:"
all_spikes = poissonsource.get_data()
for spiketrain in all_spikes.segments[0].spiketrains:
    print "cell #%d: %s" % (spiketrain.annotations['source_id'], spiketrain[:5])

poissonsource.write_data("Results/SpikeSourcePoisson_%s.pkl" % simulator_name)

end()
