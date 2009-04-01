# ==============================================================================
# Connection method classes for neuron
# $Id: connectors.py 361 2008-06-12 16:17:59Z apdavison $
# ==============================================================================

from pyNN import common
from pyNN.random import RandomDistribution, NativeRNG
from pyNN.neuron2 import simulator
import numpy
import logging
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh

# ==============================================================================
#   Utility functions/classes (not part of the API)
# ==============================================================================

class ConstIter(object):
    """An iterator that always returns the same value."""
    def __init__(self, x):
        self.x = x
    def next(self):
        return self.x

class HocConnector(object):
    
    def weights_iterator(self):
        w = self.weights
        if w is not None:
            if hasattr(w, '__len__'): # w is an array
                weights = w.__iter__()
            else:
                weights = ConstIter(w)
        else: 
            weights = ConstIter(1.0)
        return weights
    
    def delays_iterator(self):
        d = self.delays
        if d is not None:
            if hasattr(d, '__len__'): # d is an array
                if min(d) < simulator.state.min_delay:
                    raise Exception("The array of delays contains one or more values that is smaller than the simulator minimum delay.")
                delays = d.__iter__()
            else:
                delays = ConstIter(max((d, simulator.state.min_delay)))
        else:
            delays = ConstIter(simulator.state.min_delay)
        return delays

    def _process_conn_list(self, conn_list, projection):
        """Extract fields from list of tuples and construct the hoc commands."""
        for i in xrange(len(conn_list)):
            src, tgt, weight, delay = conn_list[i][:]
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            projection.connections.append(simulator.single_connect(src, tgt, weight, delay, projection.synapse_type))

def probabilistic_connect(connector, projection, p):
    weights = connector.weights_iterator()
    delays = connector.delays_iterator()
    if isinstance(projection.rng, NativeRNG):
        rarr = simulator.nativeRNG_pick(projection.pre.size*projection.post.size,
                                        projection.rng,
                                        'uniform', (0,1))
    else:
        # We use concatenate, rather than just creating
        # n=projection.pre.size*projection.post.size random numbers,
        # in case of uneven distribution of neurons between MPI nodes
        if projection.post.size > 1:
            rarr = numpy.concatenate([projection.rng.next(projection.post.size, 'uniform', (0,1)) \
                                      for src in projection.pre.all()])
        else:
            rarr = projection.rng.next(len(projection.pre), 'uniform', (0,1))
            if not hasattr(rarr, '__len__'): # rng.next(1) returns a float, not an array
                # arguably, rng.next() should return a float, rng.next(1) an array of length 1
                rarr = numpy.array([rarr])
    j = 0
    required_length = projection.pre.size * len(projection.post.local_cells) 
    assert len(rarr) >= required_length, \
           "Random array is too short (%d elements, needs %d)" % (len(rarr), required_length)
        
    create = rarr<p
    for src in projection.pre.all():
        for tgt in projection.post:    
            if connector.allow_self_connections or projection.pre != projection.post or tgt != src:
                if create[j]:
                    projection.connections.append(
                            simulator.single_connect(src, tgt,
                                                     weights.next(), delays.next(),
                                                     projection.synapse_type))
            j += 1
    assert j == required_length

# ==============================================================================
#   Connection method classes
# ==============================================================================

class AllToAllConnector(common.AllToAllConnector, HocConnector):    
    
    def connect(self, projection):
        probabilistic_connect(self, projection, 1.0)
        

class OneToOneConnector(common.OneToOneConnector, HocConnector):
    
    def connect(self, projection):
        weights = self.weights_iterator()
        delays = self.delays_iterator()
        if projection.pre.dim == projection.post.dim:
            for tgt in projection.post:
                src = tgt - projection.post.first_id + projection.pre.first_id
                projection.connections.append(
                    simulator.single_connect(src, tgt, weights.next(), delays.next(), projection.synapse_type))
        else:
            raise Exception("OneToOneConnector does not support presynaptic and postsynaptic Populations of different sizes.")


class FixedProbabilityConnector(common.FixedProbabilityConnector, HocConnector):

    def connect(self, projection):
        logging.info("Connecting %s to %s with probability %s" % (projection.pre.label,
                                                                  projection.post.label,
                                                                  self.p_connect))
        probabilistic_connect(self, projection, self.p_connect)

class DistanceDependentProbabilityConnector(common.DistanceDependentProbabilityConnector, HocConnector):
    
    def connect(self, projection):
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is True:
            dimensions = projection.post.dim
            periodic_boundaries = tuple(numpy.concatenate((dimensions, numpy.zeros(3-len(dimensions)))))
        if periodic_boundaries:
            print "Periodic boundaries activated and set to size ", periodic_boundaries
        # this is not going to work for parallel sims
        # either build up d gradually by iterating over local target cells,
        # or use the local_mask somehow to pick out only part of the distance matrix
        d = common.distances(projection.pre, projection.post, self.mask,
                             self.scale_factor, self.offset,
                             periodic_boundaries)
        p_array = eval(self.d_expression)
        if isinstance(self.weights,str):
            raise Exception("The weights distance dependent are not implemented yet")
        if isinstance(self.delays,str):
            raise Exception("The delays distance dependent are not implemented yet")
        probabilistic_connect(self, projection, p_array.flatten())

class _FixedNumberConnector(HocConnector):
    
    def _connect(self, projection, x_list, y_list, type):
        weights = self.weights_iterator()
        delays = self.delays_iterator()
      
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                raise Exception("NativeRNG not yet supported for the FixedNumberPreConnector")
            rng = projection.rng
        else:
            rng = numpy.random
        for y in y_list:            
            # pick n neurons at random
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            elif hasattr(self, 'n'):
                n = self.n
            candidates = x_list
            xs = []
            while len(xs) < n: # if the number of requested cells is larger than the size of the
                                    # presynaptic population, we allow multiple connections for a given cell
                xs += [candidates[candidates.index(id)] for id in rng.permutation(candidates)[0:n]]
                # have to use index() because rng.permutation returns ints, not ID objects
            xs = xs[:n]
            for x in xs:
                if self.allow_self_connections or (x != y):
                    if type == 'pre':
                        src = x; tgt = y  
                    elif type == 'post':
                        src = y; tgt = x
                    else:
                        raise Exception('Problem in _FixedNumberConnector')
                    projection.connections.append(
                        simulator.single_connect(src, tgt, weights.next(), delays.next(), projection.synapse_type))


class FixedNumberPreConnector(common.FixedNumberPreConnector, _FixedNumberConnector):
    
    def connect(self, projection):
        self._connect(projection, projection.pre.all_cells.flatten().tolist(), projection.post.local_cells, 'pre')


class FixedNumberPostConnector(common.FixedNumberPostConnector, _FixedNumberConnector):
     
    def connect(self, projection):
        self._connect(projection, projection.post.all_cells.flatten().tolist(), projection.pre.all_cells.flatten(), 'post')


class FromListConnector(common.FromListConnector, HocConnector):
    
    def connect(self, projection):
        self._process_conn_list(self.conn_list, projection)

    
class FromFileConnector(common.FromFileConnector, HocConnector):
    
    def connect(self, projection):
        if self.distributed:
            myid = int(h.pc.id())
            self.filename += ".%d" % myid
        # open the file...
        f = open(self.filename, 'r', 10000)
        lines = f.readlines()
        f.close()
        # gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[",1)[1]
            tgt = "[%s" % tgt.split("[",1)[1]
            input_tuples.append((eval(src), eval(tgt), float(w), float(d)))
        self._process_conn_list(input_tuples, projection)