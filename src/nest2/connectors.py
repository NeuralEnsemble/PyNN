"""
 Connection method classes for nest
 $Id$
"""

import logging
from pyNN import common, connectors
from pyNN.nest2.__init__ import nest, is_number, get_max_delay, get_min_delay
import numpy
from pyNN.random import RandomDistribution, NativeRNG, NumpyRNG
from math import *
#from random import sample
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh

CHECK_CONNECTIONS = False

class InvalidWeightError(Exception): pass

def _convertWeight(w, synapse_type):
    assert isinstance(w, numpy.ndarray), "by this point, single weights should have been transformed to arrays"
    weight = w*1000.0
    all_negative = (weight<=0).all()
    all_positive = (weight>=0).all()
    if not (all_negative or all_positive):
        raise InvalidWeightError("Weights must be either all positive or all negative")
    if synapse_type == 'inhibitory' and all_positive:
        weight *= -1
    elif synapse_type == 'excitatory':
        if not all_positive:
            raise InvalidWeightError("Weights must be positive for excitatory synapses")
    return weight

def check_connections(prj, src, intended_targets):
    conn_dict = nest.GetConnections([src], prj.plasticity_name)[0]
    if isinstance(conn_dict, dict):
        N = len(intended_targets)
        all_targets = conn_dict['targets']
        actual_targets = all_targets[-N:]
        assert actual_targets == intended_targets, "%s != %s" % (actual_targets, intended_targets)
    else:
        raise Exception("Problem getting connections for %s" % pre)



class AllToAllConnector(connectors.AllToAllConnector):    

    def connect(self, projection):
        postsynaptic_neurons  = projection.post.local_cells.flatten()
        target_list = postsynaptic_neurons.tolist()
        for pre in projection.pre.cell.flat:
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections:
                target_list = postsynaptic_neurons.tolist()
                if pre in target_list:
                    target_list.remove(pre)
            N = len(target_list)
            weights = self.get_weights(N)
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            delays = self.get_delays(N).tolist()
            projection._targets += target_list
            projection._sources += [pre]*N
            nest.DivergentConnect([pre], target_list, weights, delays, projection.plasticity_name)
            if CHECK_CONNECTIONS:
                check_connections(projection, pre, target_list)
        return len(projection._targets)

class OneToOneConnector(connectors.OneToOneConnector):
    
    def connect(self, projection):
        if projection.pre.dim == projection.post.dim:
            projection._sources = projection.pre.cell.flatten()
            projection._targets = projection.post.cell.flatten()
            N = len(projection._sources)
            weights = self.get_weights(N)
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            delays = self.get_delays(N).tolist()
            nest.Connect(projection._sources, projection._targets, weights, delays, projection.plasticity_name)
            return projection.pre.size
        else:
            raise common.InvalidDimensionsError("OneToOneConnector does not support presynaptic and postsynaptic Populations of different sizes.")

def probabilistic_connect(connector, projection, p):
    if projection.rng:
        if isinstance(projection.rng, NativeRNG):
            logging.warning("Warning: use of NativeRNG not implemented. Using NumpyRNG")
            rng = NumpyRNG()
        else:
            rng = projection.rng
    else:
        rng = NumpyRNG()
        
    local = projection.post._mask_local
    for src in projection.pre.all():
        # ( the following two lines are a nice idea, but this needs some thought for
        #   the parallel case, to ensure reproducibility when varying the number
        #   of processors
        #      N = rng.binomial(npost,self.p_connect,1)[0]
        #      targets = sample(postsynaptic_neurons, N)   # )
        N = projection.post.size
        # if running in parallel, rng.next(N) will not return N values, but only
        # as many as are needed on this node, as determined by mask_local.
        # Over the simulation as a whole (all nodes), N values will indeed be
        # returned.
        rarr = rng.next(N, 'uniform', (0, 1), mask_local=local)
        create = rarr<p
        targets = projection.post.local_cells[create].tolist()
        
        weights = connector.get_weights(N, local)[create]
        weights = _convertWeight(weights, projection.synapse_type).tolist()
        delays  = connector.get_delays(N, local)[create].tolist()
        
        if not connector.allow_self_connections and src in targets:
            assert len(targets) == len(weights) == len(delays)
            i = targets.index(src)
            weights.pop(i)
            delays.pop(i)
            targets.remove(src)
        
        projection._targets += targets
        projection._sources += [src]*len(targets)
        try:
            nest.DivergentConnect([src], targets, weights, delays, projection.plasticity_name)
        except nest.NESTError, e:
            raise common.ConnectionError("%s. src=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                         e, src, targets, weights, delays, projection.plasticity_name))
        if CHECK_CONNECTIONS:
            check_connections(projection, src, targets)
    return len(projection._sources)

class FixedProbabilityConnector(connectors.FixedProbabilityConnector):
    
    def connect(self, projection): # new implementation. Not working yet
        logging.info("Connecting %s to %s with probability %s" % (projection.pre.label,
                                                                  projection.post.label,
                                                                  self.p_connect))
        nconn = probabilistic_connect(self, projection, self.p_connect)
        return nconn
    
    
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
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            # We deal with the fact that the user could have given a delays distance dependent
            if isinstance(self.delays,str):
                delays = get_delays(distances[idx]).tolist()
            else:
                delays = self.get_delays(N).tolist()
            projection._targets += target_list
            projection._sources += [pre]*N 
            nest.DivergentConnect([pre], target_list, weights, delays, projection.plasticity_name)
            if CHECK_CONNECTIONS:
                check_connections(projection, pre, target_list)
        return len(projection._sources)

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
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            delays = self.get_delays(N).tolist()
            nest.DivergentConnect([pre], target_list, weights, delays, projection.plasticity_name)
            projection._sources += [pre]*N
            projection._targets += target_list
            if CHECK_CONNECTIONS:
                check_connections(projection, pre, target_list)
        return len(projection._sources)


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
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            delays = self.get_delays(N).tolist()

            nest.ConvergentConnect(source_list, [post], weights, delays, projection.plasticity_name)
            if CHECK_CONNECTIONS:
                for src in source_list:
                    check_connections(projection, src, [post])
            projection._sources += source_list
            projection._targets += [post]*N

        return len(projection._sources)


def _connect_from_list(conn_list, projection):
    # slow: should maybe sort by pre and use DivergentConnect?
    # or at least convert everything to a numpy array at the start
    weights = numpy.empty((len(conn_list),))
    delays = numpy.empty_like(weights)
    for i in xrange(len(conn_list)):
        src, tgt, weight, delay = conn_list[i][:]
        src = projection.pre[tuple(src)]
        tgt = projection.post[tuple(tgt)]
        projection._sources.append(src)
        projection._targets.append(tgt)
        #weights.append(_convertWeight(weight, projection.synapse_type))
        weights[i] = weight
        delays[i] = delay
    weights = _convertWeight(weights, projection.synapse_type)
    nest.Connect(projection._sources, projection._targets, weights, delays, projection.plasticity_name)
    return projection.pre.size


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