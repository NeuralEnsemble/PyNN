"""
Very simple STDP test

Andrew Davison, UNIC, CNRS
January 2008

$Id:$
"""

import sys
import numpy

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
    print simulator
#else:
#    simulator = "neuron"
    #simulator = "neuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)


setup(timestep=0.1,min_delay=0.1,max_delay=1.0,debug=True)

p1 = Population(1, SpikeSourceArray, {'spike_times': numpy.arange(0, 50, 1.0)})
p2 = Population(1, IF_curr_exp)

stdp_model = STDPMechanism(timing_dependence=SpikePairRule(tau_plus=20.0, tau_minus=20.0),
                           weight_dependence=AdditiveWeightDependence(w_min=0, w_max=0.4,
                                                                      A_plus=0.01, A_minus=0.012))

connection_method = AllToAllConnector(weights=0.24)
prj = Projection(p1, p2, method=connection_method,
                 synapse_dynamics=SynapseDynamics(slow=stdp_model))

if simulator == "nest2":
    print nest.GetConnection([p1[0]], 'stdp_synapse_hom', 0)
    print nest.GetStatus([p2[0]])

p1.record()
p2.record()
p2.record_v()
#print prj.getWeights()
t = 0
if simulator == "nest2":
    while t < 50:
        t = run(1.0)
        syn_dict = nest.GetConnection([p1[0]], 'stdp_synapse_hom', 0)
        print "%4.1f   %6.4f   %7.3f   %6.4f" % (t, prj.getWeights()[0], syn_dict['Kplus'], syn_dict['weight'])
#run(1000.0)
p1.printSpikes("simple_STDP_1.ras")
p2.printSpikes("simple_STDP_2.ras")
p2.print_v("simple_STDP.v")
#print prj.getWeights()

end()
                 
