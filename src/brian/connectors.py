# ==============================================================================
# Connection method classes for nest1
# $Id: connectors.py 294 2008-04-04 12:07:56Z apdavison $
# ==============================================================================

from pyNN import common
from pyNN.brian.__init__ import numpy
import brian_no_units_no_warnings
import brian, types
from pyNN.random import RandomDistribution, NativeRNG
from math import *
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh

def is_number(n):
    return type(n) == types.FloatType or type(n) == types.IntType or type(n) == numpy.float64

def _targetConnection(Connector, projection):
    if projection.synapse_type == "excitatory" or projection.synapse_type is None:
        target=projection.post.celltype.synapses['exc']
    else:
        target=projection.post.celltype.synapses['inh']
    src   = projection.pre.brian_cells
    tgt   = projection.post.brian_cells
    delay = max(projection.pre.brian_cells.clock.dt,Connector.delays*0.001)
    connection = brian.Connection(src, tgt, target, delay=delay)
    return connection

def _convertWeight(weight, projection):
    #if isinstance(projection.pre.brian_cells, PoissonGroupWithDelays):
        #weight *= 10*projection.pre.brian_cells.clock.dt
    if isinstance(weight, numpy.ndarray):
        all_negative = (weight<=0).all()
        all_positive = (weight>=0).all()
        assert all_negative or all_positive, "Weights must be either all positive or all negative"
        if projection.synapse_type == 'inhibitory' and all_positive:
            weight *= -1
    elif is_number(weight):
        if projection.synapse_type == 'inhibitory' and weight > 0:
            weight *= -1
    else:
        raise TypeError("we must be either a number or a numpy array")
    return weight

class AllToAllConnector(common.AllToAllConnector):    
    
    def connect(self, projection):
        projection._connections = _targetConnection(self, projection)
        weight = _convertWeight(self.weights, projection)
        projection._connections.connect_full(projection.pre.brian_cells,projection.post.brian_cells, weight=weight)
        return projection._connections.W.getnnz()

class OneToOneConnector(common.OneToOneConnector):
    
    def connect(self, projection):
        projection._connections = _targetConnection(self, projection)
        weight = _convertWeight(self.weights, projection)
        projection._connections.connect_one_to_one(projection.pre.brian_cells,projection.post.brian_cells, weight=weight)
        return projection._connections.W.getnnz()
    
class FixedProbabilityConnector(common.FixedProbabilityConnector):
    
    def connect(self, projection):
        projection._connections = _targetConnection(self, projection)
        weight = _convertWeight(self.weights, projection)
        projection._connections.connect_random(projection.pre.brian_cells,projection.post.brian_cells, self.p_connect, weight=weight)
        return projection._connections.W.getnnz()
    
class DistanceDependentProbabilityConnector(common.DistanceDependentProbabilityConnector):
    
    def connect(self, projection):
        
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is True:
            dimensions = projection.post.dim
            periodic_boundaries = tuple(numpy.concatenate((dimensions, numpy.zeros(3-len(dimensions)))))
        if periodic_boundaries:
            print "Periodic boundaries activated and set to size ", periodic_boundaries
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                print "Warning: use of NativeRNG not implemented. Using NumpyRNG"
                rng = numpy.random
            else:
                rng = projection.rng
        else:
            rng = numpy.random
        global nb_conn, post, N
        # This is just a temporary counter implemented to be sure the
        # connections are build as wanted. Because now, for the moment,
        # the connect_full method of Brian established all the connections, 
        # and not only those with non zero elements. 
        # The -2 comes from the fact that brian will make 2 tests calls before
        # using the function.
        #nb_conn = -2
        post    = projection.post
        N       = len(post)
        get_proba   = lambda d: eval(self.d_expression)
        get_weights = lambda d: eval(self.weights)
        
        def topoconnect(i,j):
            global nb_conn, post, N
            pre  = projection.pre.cell.flatten()[i]
            distances = common.distances(pre, post, self.mask,
                                         self.scale_factor, self.offset,
                                         periodic_boundaries)[0]
            p = get_proba(distances)
            # We get the list of cells that will established a connection
            rarr  = rng.uniform(0, 1, N)
            conn  = ((p >= 1) | ((0 < p) & (p < 1) & (rarr <= p)))
            if isinstance(self.weights,str):
                weights = get_weights(distances)
            else:
                weights = self.weights*numpy.ones(N)
            weights = _convertWeight(weights, projection)
            if isinstance(self.delays,str):
                raise Exception('''The string definition for delays in Brian is not 
                                implemented since Brian does not support yet 
                                heterogeneous delays''')
            if not self.allow_self_connections and conn[i]:
                conn[i] = False
            #non_conn = numpy.where(conn == False)[0]
            #nb_conn += N - len(non_conn)
            weights[numpy.equal(conn, False)] = 0.
            return weights
        projection._connections = _targetConnection(self, projection)
        projection._connections.connect_full(projection.pre.brian_cells,projection.post.brian_cells,weight=topoconnect)
        #print nb_conn
        return projection._connections.W.getnnz()
    

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