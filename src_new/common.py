# 1) instantiation of cells creates on Sim back-end

import params
import units
    


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

class Neuron(ParamObject):

    def __init__(self, params=None, syn=None):
        ParamObject.__init__(self,params)
        


NeuronType = type(Neuron())



