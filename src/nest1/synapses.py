# ==============================================================================
# Synapse Dynamics classes for nest1
# $Id:$
# ==============================================================================

from pyNN import common, synapses


class SynapseDynamics(synapses.SynapseDynamics):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Dynamic synapses are not available for this simulator.")

class STDPMechanism(synapses.STDPMechanism):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("STDP is not available for this simulator.")

class TsodkysMarkramMechanism(common.ModelNotAvailable):
    pass

class AdditiveWeightDependence(common.ModelNotAvailable):
    pass

class MultiplicativeWeightDependence(common.ModelNotAvailable):
    pass

class SpikePairRule(common.ModelNotAvailable):
    pass
