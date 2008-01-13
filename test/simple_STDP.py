"""
Very simple STDP test

Andrew Davison, UNIC, CNRS
January 2008

$Id:$
"""

import sys

if hasattr(sys,"argv"):     # run using python
    simulator = sys.argv[-1]
else:
    simulator = "neuron"
    #simulator = "oldneuron"    # run using nrngui -python


exec("from pyNN.%s import *" % simulator)
im

setup(timestep=0.1,min_delay=0.01,max_delay=1.0,debug=True)

p1 = Population(100, SpikeSourcePoisson, {'rate': 20.0})
p2 = Population(1, IF_curr_exp)

stdp_model = STDPMechanism(timing_dependence=SpikePairRule(tau_plus=20.0, tau_minus=20.0),
                           weight_dependence=AdditiveWeightDependence(w_min=0, w_max=0.2, A_plus=0.01, A_minus=0.012))

connection_method = AllToAllConnector(params={'weights': 0.1})
prj = Projection(p1, p2, method=connection_method,
                 synapse_dynamics=SynapseDynamics(slow=stdp_model))

p2.record_v()
print prj.weights()
run(1000.0)
p2.print_v("simple_STDP.v")
print prj.weights()

end()
                 