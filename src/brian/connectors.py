# ==============================================================================
# Connection method classes for nest1
# $Id: connectors.py 294 2008-04-04 12:07:56Z apdavison $
# ==============================================================================

from pyNN import common
from pyNN.brian.__init__ import numpy
import brian_no_units_no_warnings
import brian
from pyNN.random import RandomDistribution, NativeRNG
from math import *

class AllToAllConnector(common.AllToAllConnector):    
    
    def connect(self, projection):
        if projection.synapse_type == "excitatory":
            projection._connections = brian.Connection(projection.pre.cell, projection.post.cell,'ge', delay=self.delays*0.001)
        else:
            projection._connections = brian.Connection(projection.pre.cell, projection.post.cell,'gi', delay=self.delays*0.001)
        projection._connections.connect_full(projection.pre.cell,projection.post.cell, weight=self.weights)
        return projection._connections.W.getnnz()

class OneToOneConnector(common.OneToOneConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")
    
class FixedProbabilityConnector(common.FixedProbabilityConnector):
    
    def connect(self, projection):
        if projection.synapse_type == "excitatory":
            projection._connections = brian.Connection(projection.pre.cell,projection.post.cell,'ge', delay=self.delays*0.001)
        else:
            projection._connections = brian.Connection(projection.pre.cell,projection.post.cell,'gi', delay=self.delays*0.001)
        projection._connections.connect_random(projection.pre.cell,projection.post.cell, self.p_connect, weight=self.weights)
        return projection._connections.W.getnnz()
    
class DistanceDependentProbabilityConnector(common.DistanceDependentProbabilityConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")
    

class FixedNumberPreConnector(common.FixedNumberPreConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")

class FixedNumberPostConnector(common.FixedNumberPostConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")
    

class FromListConnector(common.FromListConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")
            

class FromFileConnector(common.FromFileConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")