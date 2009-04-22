# ==============================================================================
# Connection method classes for neuron
# $Id: connectors.py 361 2008-06-12 16:17:59Z apdavison $
# ==============================================================================

from pyNN import common, connectors as base_connectors
from pyNN.random import RandomDistribution, NativeRNG

import numpy
import logging
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh

# ==============================================================================
#   Utility functions/classes (not part of the API)
# ==============================================================================

class HocConnector(object):

    def _process_conn_list(self, conn_list, projection):
        #"""Extract fields from list of tuples and construct the hoc commands."""
        for i in xrange(len(conn_list)):
            src, tgt, weight, delay = conn_list[i][:]
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            projection.connection_manager.connect(src, tgt, weight, delay, projection.synapse_type)


# ==============================================================================
#   Connection method classes
# ==============================================================================


AllToAllConnector = base_connectors.AllToAllConnector
FixedProbabilityConnector = base_connectors.FixedProbabilityConnector
DistanceDependentProbabilityConnector = base_connectors.DistanceDependentProbabilityConnector
 
     
class OneToOneConnector(base_connectors.OneToOneConnector):
    
    def connect(self, projection):
        weights = self.weights_iterator()
        delays = self.delays_iterator()
        if projection.pre.dim == projection.post.dim:
            for tgt in projection.post:
                src = tgt - projection.post.first_id + projection.pre.first_id
                projection.connection_manager.connect(src, tgt, weights.next(), delays.next(), projection.synapse_type)
        else:
            raise Exception("OneToOneConnector does not support presynaptic and postsynaptic Populations of different sizes.")


class _FixedNumberConnector(object):
    
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
                    projection.connection_manager.connect(src, tgt, weights.next(), delays.next(), projection.synapse_type)


class FixedNumberPreConnector(base_connectors.FixedNumberPreConnector, _FixedNumberConnector):
    
    def connect(self, projection):
        self._connect(projection, projection.pre.all_cells.flatten().tolist(), projection.post.local_cells, 'pre')


class FixedNumberPostConnector(base_connectors.FixedNumberPostConnector, _FixedNumberConnector):
     
    def connect(self, projection):
        self._connect(projection, projection.post.all_cells.flatten().tolist(), projection.pre.all_cells.flatten(), 'post')


class FromListConnector(base_connectors.FromListConnector, HocConnector):
    
    def connect(self, projection):
        self._process_conn_list(self.conn_list, projection)

    
class FromFileConnector(base_connectors.FromFileConnector, HocConnector):
    
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