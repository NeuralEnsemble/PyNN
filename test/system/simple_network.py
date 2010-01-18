# encoding: utf-8
"""
Defines a very simple network - one spike source connected to one post-synaptic
neuron - encapsulated in the `SimpleNetwork` class. The connections may be
plastic, depending on the parameterization.

The network may be extended using the `add_population()` and `add_projection()`
methods.

Andrew Davison, UNIC, CNRS, 2008
"""

import numpy
from NeuroTools.parameters import ParameterSet
from NeuroTools import signals
from itertools import chain

class SimpleNetwork(object):

    required_parameters = ParameterSet({
        'system': dict,
        'input_spike_times': numpy.ndarray,
        'cell_type': str,
        'cell_parameters': dict,
        'plasticity': dict,
        'weights': float,
        'delays': float
    })

    @classmethod
    def check_parameters(cls, parameters):
        assert isinstance(parameters, ParameterSet)
        for name, p_type in cls.required_parameters.flat():
            assert isinstance(parameters[name], p_type), "%s: expecting %s, got %s" % (name, p_type, type(parameters[name]))
        return True

    def __init__(self, sim, parameters):
        self.sim = sim
        self.parameters = parameters
        sim.setup(**parameters.system)
        # Create cells
        self.pre = sim.Population(1, sim.SpikeSourceArray,
                                  {'spike_times': parameters.input_spike_times},
                                  label='pre')
        self.post = sim.Population(1, getattr(sim, parameters.cell_type),
                                   parameters.cell_parameters,
                                   label='post')
        self._source_populations = set([self.pre])
        self._neuronal_populations = set([self.post])
        # Create synapse model
        if parameters.plasticity.short_term:
            P = parameters.plasticity.short_term
            fast_mech = getattr(sim, P.model)(P.parameters)
        else:
            fast_mech = None
        if parameters.plasticity.long_term:
            P = parameters.plasticity.long_term
            print P.pretty()
            print P.timing_dependence
            print P.timing_dependence.model
            print P.timing_dependence.params
            slow_mech = sim.STDPMechanism(
                timing_dependence=getattr(sim, P.timing_dependence.model)(**P.timing_dependence.params),
                weight_dependence=getattr(sim, P.weight_dependence.model)(**P.weight_dependence.params),
                dendritic_delay_fraction=P.ddf
            )
        else:
            slow_mech = None
        if fast_mech or slow_mech:
            syn_dyn = sim.SynapseDynamics(fast=fast_mech, slow=slow_mech)
        else:
            syn_dyn = None
        # Create connections
        weights = parameters.weights          # to give approximate parity in numerical values between
        if "_cond_" in parameters.cell_type:  # current- and conductance-based synapses, we divide the
            weights /= 50.0                   # weights by 50 (approx the distance in mV between threshold
                                              # and excitatory reversal potential) for conductance-based.
        connector = sim.AllToAllConnector(weights=weights,
                                          delays=parameters.delays)
        self.prj = [sim.Projection(self.pre, self.post, method=connector,
                                   synapse_dynamics=syn_dyn)]
        # Setup recording
        self.pre.record()
        self.post.record()
        self.post.record_v()
        
    def add_population(self, label, dim, cell_type, parameters={}):
        cell_type = getattr(self.sim, cell_type)
        pop = self.sim.Population(dim, cell_type, parameters, label=label)
        setattr(self, label, pop)
        pop.record()
        if cell_type.synapse_types: # don't record Vm for populations that don't have it
            pop.record_v()
            self._neuronal_populations.add(pop)
        else:
            self._source_populations.add(pop)
        
    def add_projection(self, src, tgt, connector_name, connector_parameters={}):
        connector = getattr(self.sim, connector_name)(**connector_parameters)
        assert hasattr(self, src)
        assert hasattr(self, tgt)
        prj = self.sim.Projection(getattr(self, src), getattr(self, tgt), connector)
        self.prj.append(prj)
    
    def save_weights(self):
        t = self.sim.get_current_time()
        for prj in self.prj:
            w = prj.getWeights()
            w.insert(0, t)
            prj.weights.append(w)
    
    def get_spikes(self):
        spikes = {}
        for pop in chain(self._neuronal_populations, self._source_populations):
            spike_arr = pop.getSpikes()
            spikes[pop.label] = signals.SpikeList(spike_arr, id_list=range(pop.size))
        return spikes
        
    def get_v(self):
        vm = {}
        for pop in self._neuronal_populations:
            vm_arr = pop.get_v()
            vm[pop.label] = signals.VmList(vm_arr[:,(0,2)], id_list=range(pop.size),
                                           dt=self.parameters.system.timestep,
                                           t_start=min(vm_arr[:,1]),
                                           t_stop=max(vm_arr[:,1])+self.parameters.system.timestep)
        return vm
    
    def get_weights(self, at_input_spiketimes=False, recording_interval=1.0):
        w = {}
        if at_input_spiketimes:
            presynaptic_spike_times = self.pre.getSpikes()[:,1]
        for prj in self.prj:
            w[prj.label] = numpy.array(prj.weights)
            if at_input_spiketimes:
                assert isinstance(prj._method.delays, float)
                post_synaptic_potentials = presynaptic_spike_times + prj._method.delays
                mask = numpy.ceil(post_synaptic_potentials/recording_interval).astype('int')
                w[prj.label] = w[prj.label][mask]
        return w

        
def test(sim):
    
    params = ParameterSet({
        'system': { 'timestep': 0.1, 'min_delay': 0.1, 'max_delay': 10.0 },
        'input_spike_times': numpy.arange(5,105,10.0),
        'cell_type': 'IF_curr_exp',
        'cell_parameters': {},
        'plasticity': { 'short_term': None, 'long_term': None },
        'weights': 0.1,
        'delays': 1.0,
    })
    SimpleNetwork.check_parameters(params)
    net = SimpleNetwork(sim, params)
    sim.run(100.0)
    id = net.get_v()['post'].id_list()[0]
    print id
    print net.get_v()['post'][id]
    
        
# ==============================================================================
if __name__ == "__main__":
    import pyNN.nest
    test(pyNN.nest)
    