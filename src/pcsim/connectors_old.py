# ==============================================================================
# Connection method classes for pcsim
# ==============================================================================

from pyNN import common, connectors
import pypcsim
from pyNN.pcsim import simulator
import numpy


class ListConnectionPredicate(pypcsim.PyConnectionDecisionPredicate):
    """Used by FromListConnector and FromFileConnector."""
    
    def __init__(self, conn_array):
        pypcsim.PyConnectionDecisionPredicate.__init__(self)
        # now need to turn conn_list into a form suitable for use by decide()
        # a sparse array would be one possibility, but for now we use a dict of dicts
        self._connections = {}
        for i in xrange(len(conn_array)):
            src, tgt = conn_array[i][:]
            if src not in self._connections:
                self._connections[src] = []
            self._connections[src].append(tgt) 
    
    def decide(self, src, tgt, rng):
        if src in self._connections and tgt in self._connections[src]:
            return True
        else:
            return False


class AllToAllConnector(connectors.AllToAllConnector):    
    
    def connect(self, projection):
        
        # what about allow_self_connections?
        decider = pypcsim.RandomConnections(1)
        wiring_method = pypcsim.DistributedSyncWiringMethod(simulator.net)
        return decider, wiring_method, self.weights, self.delays

class OneToOneConnector(connectors.OneToOneConnector):
    
    def connect(self, projection):
        
        if projection.pre.dim == projection.post.dim:
            decider = pypcsim.RandomConnections(1)
            wiring_method = pypcsim.OneToOneWiringMethod(simulator.net)
            return decider, wiring_method, self.weights, self.delays
        else:
            raise Exception("Connection method not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes.")

class FixedProbabilityConnector(connectors.FixedProbabilityConnector):
    
    def connect(self, projection):
        
        decider = pypcsim.RandomConnections(float(self.p_connect))
        wiring_method = pypcsim.DistributedSyncWiringMethod(simulator.net)
        return decider, wiring_method, self.weights, self.delays

#class DistanceDependentProbabilityConnector(connectors.DistanceDependentProbabilityConnector):
#    
#    def connect(self, projection):
#        decider = pypcsim.EuclideanDistanceRandomConnections(method_parameters[0], method_parameters[1]) 
#        wiring_method = pypcsim.DistributedSyncWiringMethod(simulator.net)
#        return decider, wiring_method, self.weights, self.delays

class FixedNumberPreConnector(connectors.FixedNumberPreConnector):
    
    def connect(self, projection):
        
        decider = pypcsim.DegreeDistributionConnections(pypcsim.ConstantNumber(self.n), pypcsim.DegreeDistributionConnections.incoming)
        wiring_method = pypcsim.SimpleAllToAllWiringMethod(simulator.net)
        return decider, wiring_method, self.weights, self.delays

class FixedNumberPostConnector(connectors.FixedNumberPostConnector):
    
    def connect(self, projection):
        decider = pypcsim.DegreeDistributionConnections(pypcsim.ConstantNumber(self.n), pypcsim.DegreeDistributionConnections.outgoing)
        wiring_method = pypcsim.SimpleAllToAllWiringMethod(simulator.net)
        return decider, wiring_method, self.weights, self.delays
    
class FromListConnector(connectors.FromListConnector):
    
    def connect(self, projection):
        conn_array = numpy.zeros((len(self.conn_list),4))
        for i in xrange(len(self.conn_list)):
            src, tgt, weight, delay = self.conn_list[i][:]
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            conn_array[i,:] = (src, tgt, weight, delay)
        self.weights = conn_array[:,2]
        self.delays = conn_array[:,3]
        lcp = ListConnectionPredicate(conn_array[:,0:2])
        decider = pypcsim.PredicateBasedConnections(lcp)
        wiring_method = pypcsim.SimpleAllToAllWiringMethod(simulator.net)
        # pcsim does not yet deal with having lists of weights, delays, so for now we just return 0 values
        # and will set the weights, delays later
        return decider, wiring_method, self.weights, self.delays

class FromFileConnector(connectors.FromFileConnector):
    
    def connect(self, projection):
        f = open(self.filename, 'r', 10000)
        lines = f.readlines()
        f.close()
        conn_array = numpy.zeros((len(lines),4))
        for i,line in enumerate(lines):
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[",1)[1]
            tgt = "[%s" % tgt.split("[",1)[1]
            src = projection.pre[tuple(eval(src))]
            tgt = projection.post[tuple(eval(tgt))]
            conn_array[i,:] = (src, tgt, w, d)
        self.weights = conn_array[:,2]
        self.delays = conn_array[:,3]
        lcp = ListConnectionPredicate(conn_array[:,0:2])
        decider = pypcsim.PredicateBasedConnections(lcp)
        wiring_method = pypcsim.SimpleAllToAllWiringMethod(simulator.net)
        # pcsim does not yet deal with having lists of weights, delays, so for now we just return 0 values
        # and will set the weights, delays later
        return decider, wiring_method, self.weights, self.delays
