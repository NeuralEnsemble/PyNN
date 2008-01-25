# 1) instantiation of cells creates on Sim back-end

import params
import units
    
# recordables
# v, g_e,g_i
# class Neuron(HasRecordables)
# 

# weight dependence, {all-to-all, nearest neighbor}, {pair,triplet}
class SynapsePlasticity(ParamObject):

    def __init__(self,params = None):
        ParamObject.__init__(self,params)





class SynapseDynamics(ParamObject):

    def __init__(self,params=None):
        ParamObject.__init__(self,params)




# Synapses

# cond_exp, curr_alpha, etc.
class Synapse(ParamObject):

    def __init__(self, params=None, dynamics=None,plasticity=None):
        ParamObject.__init__(self,params)



SynapseType = type(Synapse())



SynapseDynamicsType = type(SynapseDynamics())

# adaptation, refractoriness, triplet filters
class NeuronDynamics(ParamObject):

    def __init__(self, params=None):
        ParamObject.__init__(self,params)
        

class Neuron(ParamObject):

    def __init__(self, params=None, syn=None, dyn=None):
        ParamObject.__init__(self,params)
        # dyn is for things like adaptation

    def pre(self,post,syn=None):
        """make self presynaptic to post with synapse type syn."""
        # NEST backend cannot specify syn
        # syn must be specified at init
        # many exceptions: selective synapses
        # NEST:cond, cur must be secified upon neuron creation
        # PCSIM: can be specified after fact?
        # NEST: 1 syn dyn can be specified after the fact
        # Multiple syn dyns?: stdp, syn dyn

        pass

    def post(self,pre,syn=None):
        """make self postsynaptic to pre with synapse type syn."""
        # see pre 
        pass
    
    


NeuronType = type(Neuron())



