"""
Connection method classes for nest

$Id$
"""
from pyNN import random, common

from pyNN.connectors import AllToAllConnector, \
                            OneToOneConnector, \
                            FixedProbabilityConnector, \
                            DistanceDependentProbabilityConnector, \
                            FromListConnector, \
                            FromFileConnector, \
                            FixedNumberPreConnector, \
                            FixedNumberPostConnector


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
