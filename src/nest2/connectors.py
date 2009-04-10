"""
 Connection method classes for nest
 $Id$
"""

import logging
import nest
from pyNN import common, connectors
import numpy
from pyNN.random import RandomDistribution, NativeRNG, NumpyRNG
from math import *
#from random import sample
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh

CHECK_CONNECTIONS = False


def check_connections(prj, src, intended_targets):
    conn_dict = nest.GetConnections([src], prj.plasticity_name)[0]
    if isinstance(conn_dict, dict):
        N = len(intended_targets)
        all_targets = conn_dict['targets']
        actual_targets = all_targets[-N:]
        assert actual_targets == intended_targets, "%s != %s" % (actual_targets, intended_targets)
    else:
        raise Exception("Problem getting connections for %s" % pre)


# ==============================================================================
#   Connection method classes
# ==============================================================================

AllToAllConnector = connectors.AllToAllConnector
FixedProbabilityConnector = connectors.FixedProbabilityConnector

class OneToOneConnector(connectors.OneToOneConnector):
    
    def connect(self, projection):
        if projection.pre.dim == projection.post.dim:
            N = projection.post.size
            local = projection.post._mask_local
            weights = self.get_weights(N, local)
            is_conductance = common.is_conductance(projection.post.index(0))
            weights = common.check_weight(weights, projection.synapse_type, is_conductance)
            delays = self.get_delays(N, local)
            for src, tgt, w, d in zip(projection.pre.local_cells,
                                      projection.post.local_cells,
                                      weights,
                                      delays):
                # the float is in case the values are of type numpy.float64, which NEST chokes on
                projection.connection_manager.connect([src], [tgt], float(w), float(d), projection.synapse_type)
                ##nest.Connect(projection._sources, projection._targets, weights, delays, projection.plasticity_name)
        else:
            raise common.InvalidDimensionsError("OneToOneConnector does not support presynaptic and postsynaptic Populations of different sizes.")

    
    
class DistanceDependentProbabilityConnector(connectors.DistanceDependentProbabilityConnector):
    
    def connect(self, projection):
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is True:
            dimensions = projection.post.dim
            periodic_boundaries = tuple(numpy.concatenate((dimensions, numpy.zeros(3-len(dimensions)))))
        if periodic_boundaries:
            logging.info("Periodic boundaries activated and set to size %s" % (periodic_boundaries,))
        postsynaptic_neurons = projection.post.cell.flatten() # array
        npost = len(postsynaptic_neurons)
        #postsynaptic_neurons = projection.post.local_cells
        # what about NativeRNG?
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                logging.warning("Warning: use of NativeRNG not implemented. Using NumpyRNG")
                rng = numpy.random
            else:
                rng = projection.rng
        else:
            rng = numpy.random
            
        get_proba   = lambda d: eval(self.d_expression)
        get_weights = lambda d: eval(self.weights)
        get_delays  = lambda d: eval(self.delays)
        is_conductance = common.is_conductance(projection.post.index(0))
        
        for pre in projection.pre.cell.flat:
            # We compute the distances from the post cell to all the others
            distances = common.distances(pre, projection.post, self.mask,
                                         self.scale_factor, self.offset,
                                         periodic_boundaries)[0]
            # We evaluate the probabilities of connections for those distances
            proba = get_proba(distances)
            # We get the list of cells that will established a connection
            rarr = rng.uniform(0, 1, (npost,))
            idx = numpy.where((proba >= 1) | ((0 < proba) & (proba < 1) & (rarr <= proba)))[0]
            target_list = postsynaptic_neurons[idx].tolist()
            # We remove the pre cell if we don't allow self connections
            if not self.allow_self_connections and pre in target_list:
                idx = numpy.delete(idx, target_list.index(pre))
                target_list.remove(pre)
            N = len(target_list)
            
            # We deal with the fact that the user could have given a weights distance dependent
            if isinstance(self.weights,str):
                weights = get_weights(distances[idx])
            else:
                weights = self.get_weights(N)
            
            weights = common.check_weight(weights, projection.synapse_type, is_conductance)
            # We deal with the fact that the user could have given a delays distance dependent
            if isinstance(self.delays,str):
                delays = get_delays(distances[idx])
            else:
                delays = self.get_delays(N)
                
            if len(target_list) > 0:
                projection.connection_manager.connect(pre, target_list, weights, delays, projection.synapse_type)

            if CHECK_CONNECTIONS:
                check_connections(projection, pre, target_list)
        

class FixedNumberPostConnector(connectors.FixedNumberPostConnector):
    
    def connect(self, projection):
        postsynaptic_neurons  = projection.post.cell.flatten()
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                logging.warning("Warning: use of NativeRNG not implemented. Using NumpyRNG")
                rng = numpy.random
            else:
                rng = projection.rng
        else:
            rng = numpy.random
        for pre in projection.pre.cell.flat:
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
                assert n > 0
                
            if not self.allow_self_connections and projection.pre == projection.post:
                # if self connections are not allowed, remove `post` from the target list before picking the n values
                tmp_postsyn = postsynaptic_neurons.tolist()
                tmp_postsyn.remove(pre)
                target_list = rng.permutation(tmp_postsyn)[0:n].tolist()   
            else:
                target_list = rng.permutation(postsynaptic_neurons)[0:n].tolist()

            N = len(target_list)
            weights = self.get_weights(N)
            is_conductance = common.is_conductance(projection.post.index(0))
            weights = common.check_weight(weights, projection.synapse_type, is_conductance)
            delays = self.get_delays(N)
            projection.connection_manager.connect(pre, target_list, weights, delays, projection.synapse_type)

            if CHECK_CONNECTIONS:
                check_connections(projection, pre, target_list)


class FixedNumberPreConnector(connectors.FixedNumberPreConnector):
    
    def connect(self, projection):
        presynaptic_neurons = projection.pre.cell.flatten()
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                logging.warning("Warning: use of NativeRNG not implemented. Using NumpyRNG")
                rng = numpy.random
            else:
                rng = projection.rng
        else:
            rng = numpy.random
        for post in projection.post.cell.flat:
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
                
            if not self.allow_self_connections and projection.pre == projection.post:
                # if self connections are not allowed, remove `post` from the source list before picking the n values
                tmp_presyn = presynaptic_neurons.tolist()
                tmp_presyn.remove(post)
                source_list = rng.permutation(tmp_presyn)[0:n].tolist()    
            else:
                source_list = rng.permutation(presynaptic_neurons)[0:n].tolist()
            
            N = len(source_list)
            weights = self.get_weights(N)
            is_conductance = common.is_conductance(projection.post.index(0))
            weights = common.check_weight(weights, projection.synapse_type, is_conductance)
            delays = self.get_delays(N)
            
            for source, w, d in zip(source_list, weights, delays):
                projection.connection_manager.connect(source, [post], w, d, projection.synapse_type)
                if CHECK_CONNECTIONS:
                    check_connections(projection, source, [post])


def _connect_from_list(conn_list, projection):
    # slow: should maybe sort by pre and use DivergentConnect?
    # or at least convert everything to a numpy array at the start
    weights = numpy.empty((len(conn_list),))
    delays = numpy.empty_like(weights)
    for i in xrange(len(conn_list)):
        src, tgt, weight, delay = conn_list[i][:]
        src = projection.pre[tuple(src)]
        tgt = projection.post[tuple(tgt)]
        projection.connection_manager.connect(src, [tgt], weight, delay, projection.synapse_type)

class FromListConnector(connectors.FromListConnector):
    
    def connect(self, projection):
        return _connect_from_list(self.conn_list, projection)


class FromFileConnector(connectors.FromFileConnector):
    
    def connect(self, projection):
        f = open(self.filename, 'r', 10000)
        lines = f.readlines()
        f.close()
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[",1)[1]
            tgt = "[%s" % tgt.split("[",1)[1]
            input_tuples.append((eval(src), eval(tgt), float(w), float(d)))
        return _connect_from_list(input_tuples, projection)