# ==============================================================================
# Connection method classes for pcsim
# $Id$
# ==============================================================================

from pyNN import common
from pypcsim import *

class AllToAllConnector(common.AllToAllConnector):    
    
    def connect(self, projection):
        
        # what about allow_self_connections?
        decider = RandomConnections(1)
        wiring_method = DistributedSyncWiringMethod(pcsim_globals.net)
        return decider, wiring_method, self.weights, self.delays

class OneToOneConnector(common.OneToOneConnector):
    
    def connect(self, projection):
        
        if projection.pre.dim == projection.post.dim:
            decider = RandomConnections(1)
            wiring_method = OneToOneWiringMethod(pcsim_globals.net)
            return decider, wiring_method, self.weights, self.delays
        else:
            raise Exception("Connection method not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes.")

class FixedProbabilityConnector(common.FixedProbabilityConnector):
    
    def connect(self,projection):
        
        decider = RandomConnections(float(self.p_connect))
        wiring_method = DistributedSyncWiringMethod(pcsim_globals.net)
        return decider, wiring_method, self.weights, self.delays

class FixedNumberPreConnector(common.FixedNumberPreConnector):
    
    def connect(self, projection):
        
        decider = DegreeDistributionConnections(ConstantNumber(self.fixedpre), DegreeDistributionConnections.incoming)
        wiring_method = SimpleAllToAllWiringMethod(pcsim_globals.net)
        return decider, wiring_method, self.weights, self.delays

class FixedNumberPostConnector(common.FixedNumberPostConnector):
    
    def connect(self, projection):
        decider = DegreeDistributionConnections(ConstantNumber(self.fixedpost), DegreeDistributionConnections.outgoing)
        wiring_method = SimpleAllToAllWiringMethod(pcsim_globals.net)
        return decider, wiring_method, self.weights, self.delays