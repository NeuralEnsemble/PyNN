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
#   Connection method classes
# ==============================================================================


AllToAllConnector = base_connectors.AllToAllConnector

FixedProbabilityConnector = base_connectors.FixedProbabilityConnector

DistanceDependentProbabilityConnector = base_connectors.DistanceDependentProbabilityConnector

FromListConnector = base_connectors.FromListConnector

FromFileConnector = base_connectors.FromFileConnector

OneToOneConnector = base_connectors.OneToOneConnector


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



