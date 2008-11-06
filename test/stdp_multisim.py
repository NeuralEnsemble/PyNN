"""
Comparison of STDP in `neuron` and `nest2` modules.

A 'trigger' population is connected to the output population with static
synapses and very large weights, so that a single presynaptic spike produces a
single postsynaptic spike.
An 'early' population produces presynaptic spikes a short time before each
postsynaptic spike and a 'late' population a short time after.
Both are connected with very weak, plastic synapses, so as not to affect the
post-synaptic spike time.

The connections from the 'early' population should potentiate (pre before post)
and those from the 'late' population depress (post before pre).

Note that NEURON updates weight values with every pre- or post-synaptic spike,
while NEST only updates the values with every pre-synaptic spike. This gives
identical results, since weight values are only actually used when there is a
presynaptic spike, but it makes plotting/recording the weights more difficult.

Andrew Davison, UNIC, CNRS
January 2008

$Id$
"""

import sys
import numpy

from pyNN import nest2, neuron, pcsim
from NeuroTools.stgen import StGen
from NeuroTools.analysis import arrays_almost_equal

simulators = (nest2, neuron, pcsim)

# Parameters
spike_interval = 30.0
tstop = 300 #spike_interval*3
recording_interval = 1.0
first_spike = 12.0
early_offset = 7.0
late_offset = 7.0
ddf = 1.0
w_max = 0.4
delays = 2.0
acceptable_jitter = 0.03
noisy = False

assert early_offset/spike_interval < 0.5
assert late_offset/spike_interval < 0.5

def create_network(sim):
    """Create the network described above."""
    spike_latency = 0.20
    sim.setup(timestep=0.01, min_delay=0.1, max_delay=10.0,
              debug=True, quit_on_end=False)
    if noisy:
        trigger_times = StGen().poisson_generator(1.0/spike_interval, tstop) + first_spike
    else:
        trigger_times = numpy.arange(first_spike, tstop, spike_interval)
    p_early = sim.Population(1, sim.SpikeSourceArray,
                             {'spike_times': trigger_times - early_offset})
    p_late = sim.Population(1, sim.SpikeSourceArray,
                            {'spike_times': trigger_times + late_offset})
    p_tr = sim.Population(1, sim.SpikeSourceArray,
                          {'spike_times': trigger_times - spike_latency})
    p_out = sim.Population(1, sim.IF_curr_exp,
                           {'tau_syn_E': 0.5, 'tau_m': 2.0, 'cm': 0.1, 'tau_refrac': 2.0})

    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=w_max,
                                                       A_plus=0.01, A_minus=0.01),
        dendritic_delay_fraction=ddf
        )

    connector_in_out = sim.AllToAllConnector(weights=w_max/4.0, delays=delays)
    connector_tr_out = sim.AllToAllConnector(weights=10.0, delays=delays)
    prj_early_out = sim.Projection(p_early, p_out, method=connector_in_out,
                                   synapse_dynamics=sim.SynapseDynamics(slow=stdp_model))
    prj_late_out = sim.Projection(p_late, p_out, method=connector_in_out,
                                  synapse_dynamics=sim.SynapseDynamics(slow=stdp_model))
    prj_tr_out = sim.Projection(p_tr, p_out, method=connector_tr_out,
                                synapse_dynamics=None)

    p_early.record()
    p_late.record()
    p_tr.record()
    p_out.record()
    p_out.record_v()
        
    return locals()

# Create the networks
net = {}
for sim in simulators:
    net[sim.__name__] = create_network(sim)

# Create lists to store weights
prj_list = ['prj_early_out', 'prj_late_out']
weights = {}
for sim_name in net.keys():
    weights[sim_name] = {}
    for prj in prj_list:
        weights[sim_name][prj] = []

# Run the network, and record weights at short intervals
t = 0      
while t < tstop:
    for sim in simulators:
        t = sim.run(recording_interval)
        for prj in prj_list:
            weights[sim.__name__][prj].append(net[sim.__name__][prj].getWeights()[0])

# Plot weights
import pylab
pylab.rcParams['interactive'] = True
for prj in prj_list:
    for sim_name in net.keys():    
        pylab.plot(weights[sim_name][prj], label="%s %s" % (sim_name, prj))
pylab.legend(loc='center right')

# Plot spikes on same axes
spikes = {}
for p,x in (('p_early',-0.02), ('p_late',-0.03), ('p_out',-0.025)):
    spikes[p] = {}
    for sim_name in net.keys():
        spikes[p][sim_name] = net[sim_name][p].getSpikes()[:,1]
    assert arrays_almost_equal(spikes[p][nest2.__name__], spikes[p][neuron.__name__], acceptable_jitter), "Max diff = %g ms" % max(spikes[p][nest2.__name__]-spikes[p][neuron.__name__])
    assert arrays_almost_equal(spikes[p][nest2.__name__], spikes[p][pcsim.__name__], acceptable_jitter), "Max diff = %g ms" % max(spikes[p][nest2.__name__]-spikes[p][neuron.__name__])
    spike_times = spikes[p][neuron.__name__]
    pylab.plot(spike_times,
               [x*w_max]*len(spike_times), '+', label=p)
pylab.xlabel('Time (ms)')
pylab.ylabel('Weight (nA)')
pylab.title('stdp_multisim')

# plot Vm
pylab.figure(2)
for sim_name in net.keys():
    print sim_name
    vm = net[sim_name]['p_out'].get_v()
    pylab.plot(vm[:,1], vm[:,2], label=sim_name)
pylab.legend(loc='upper right')

# Extract weights at times of presynaptic spikes.
weights_at_spiketimes = {'prj_early_out': {}, 'prj_late_out': {}}
presynaptic_spikes = {}
post_synaptic_potentials = {}
for sim_name in net.keys():
    for p_name, prj_name in zip(('p_early', 'p_late'), weights_at_spiketimes.keys()):
        presynaptic_spikes[sim_name] = spikes[p_name][sim_name]
        print "presynaptic spikes for %s: %s" % (sim_name, presynaptic_spikes[sim_name])
        post_synaptic_potentials[sim_name] = presynaptic_spikes[sim_name] + delays
        #mask = (presynaptic_spikes[sim_name]/recording_interval).astype('int')
        mask = (post_synaptic_potentials[sim_name]/recording_interval).astype('int')
        weights_at_spiketimes[prj_name][sim_name] = numpy.array(weights[sim_name][prj_name])[mask]

def all_equal(arr, threshold=0):
    equal = True
    a = arr[0]
    for b in arr[1:]:
        equal = equal and (numpy.abs(a - b) <= threshold).all()
        print "---", numpy.abs(a-b)
    return equal

if (all_equal(weights_at_spiketimes['prj_early_out'].values(), threshold=1e-5) and
    all_equal(weights_at_spiketimes['prj_late_out'].values(), threshold=1e-5)):
    print "*** Passed ***"
else:
    print "*** Failed ***"
for sim_name in net.keys():
    print "%s weights: %s %s" % (sim_name,
                                 weights_at_spiketimes['prj_early_out'][sim_name],
                                 weights_at_spiketimes['prj_late_out'][sim_name])

for sim in simulators:
    sim.end()
