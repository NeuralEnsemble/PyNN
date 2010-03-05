"""
Connection method classes for nest

$Id$
"""
from pyNN import random, common, core
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh
from pyNN.connectors import AllToAllConnector, \
                            OneToOneConnector, \
                            FixedProbabilityConnector, \
                            DistanceDependentProbabilityConnector, \
                            FromListConnector, \
                            FromFileConnector, \
                            FixedNumberPreConnector, \
                            FixedNumberPostConnector
import numpy

class FixedNumberPreConnector(FixedNumberPreConnector):
    __doc__ = FixedNumberPreConnector.__doc__
    #we over-ride connect() so as to use convergent_connect()
    
    def connect(self, projection):
        """Connect-up a Projection."""
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Warning: use of NativeRNG not implemented.")
            
        for target in projection.post.local_cells.flat:
            # pick n neurons at random
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n

            candidates = projection.pre.all_cells.flatten().tolist()
            if not self.allow_self_connections and projection.pre == projection.post:
                candidates.remove(target)
            sources = []
            while len(sources) < n: # if the number of requested cells is larger than the size of the
                                    # presynaptic population, we allow multiple connections for a given cell
                sources += [candidates[candidates.index(id)] for id in projection.rng.permutation(candidates)[0:n]]
                # have to use index() because rng.permutation returns ints, not ID objects
            sources = sources[:n]
            
            weights = self.get_weights(n)
            is_conductance = common.is_conductance(projection.post.index(0))
            weights = common.check_weight(weights, projection.synapse_type, is_conductance)
            delays = self.get_delays(n)
            
            projection.connection_manager.convergent_connect(sources, [target], weights, delays)


class DistanceDependentProbabilityConnector(DistanceDependentProbabilityConnector):

    __doc__ = DistanceDependentProbabilityConnector.__doc__
    
    def connect(self, projection):
        """Connect-up a Projection."""
        local          = projection.post._mask_local.flatten()
        positions      = self.space.scale_factor*(projection.post.positions[:, local] + self.space.offset)
        N              = projection.post.size            
        is_conductance = common.is_conductance(projection.post.index(0))   
        
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Use of NativeRNG not implemented.")
        else:
            rng = projection.rng
        
        for src in projection.pre.all():            
            d     = self.space.distances(src.position, positions).flatten()
            proba = eval(self.d_expression)
            if proba.dtype == 'bool':
                proba = proba.astype(float)
            rarr = rng.next(N, 'uniform', (0, 1), mask_local=local)
            if not core.is_listlike(rarr) and numpy.isscalar(rarr): # if N=1, rarr will be a single number
                rarr = numpy.array([rarr])
            create = rarr < proba
            if create.shape != projection.post.local_cells.shape:
                logger.warning("Too many random numbers. Discarding the excess. Did you specify MPI rank and number of processes when you created the random number generator?")
                create = create[:projection.post.local_cells.size]
            targets = projection.post.local_cells[create].tolist()
            
            if hasattr(self, 'w_expr'):
                weights = eval(self.w_expr)[create]
            else:
                weights = self.get_weights(N, local)[create]            
            weights = common.check_weight(weights, projection.synapse_type, is_conductance)
            
            if hasattr(self, 'd_expr'):
                delays = eval(self.d_expr)[create]
            else:
                delays = self.get_delays(N, local)[create]            
                
            if not self.allow_self_connections and projection.pre == projection.post and src in targets:
                assert len(targets) == len(weights) == len(delays)
                i       = targets.index(src)
                weights = numpy.delete(weights, i)
                delays  = numpy.delete(delays, i)
                targets.remove(src)

            if len(targets) > 0:
                projection.connection_manager.connect(src, targets, weights, delays)