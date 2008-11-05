
import numpy
from NeuroTools.parameters import ParameterSet

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
                                  label='presynaptic cell')
        self.post = sim.Population(1, getattr(sim, parameters.cell_type),
                                   parameters.cell_parameters,
                                   label='postsynaptic cell')
        # Create synapse model
        if parameters.plasticity.short_term:
            P = parameters.plasticity.short_term
            fast_mech = getattr(sim, P.model)(P.parameters)
        else:
            fast_mech = None
        if parameters.plasticity.long_term:
            P = parameters.plasticity.long_term
            slow_mech = sim.STDPMechanism(
                timing_dependence=getattr(sim, P.timing_dependence.model)(P.timing_dependence.parameters),
                weight_dependence=getattr(sim, P.weight_dependence.model)(P.weight_dependence.parameters),
                dendritic_delay_fraction=P.ddf
            )
        else:
            slow_mech = None
        if fast_mech or slow_mech:
            syn_dyn = sim.SynapseDynamics(fast=fast_mech, slow=slow_mech)
        else:
            syn_dyn = None
        # Create connections
        connector = sim.AllToAllConnector(weights=parameters.weights,
                                          delays=parameters.delays)
        self.prj = sim.Projection(self.pre, self.post, method=connector,
                                  synapse_dynamics=syn_dyn)
        
def test():
    import pyNN.nest2
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
    net = SimpleNetwork(pyNN.nest2, params)
    
        
# ==============================================================================
if __name__ == "__main__":
    test()
    