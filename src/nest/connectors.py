"""
Connection method classes for nest

$Id$
"""
from pyNN import random, common, core
from pyNN.connectors import *
import numpy
from pyNN.space import Space

class FixedNumberPreConnector(FixedNumberPreConnector):
    
    __doc__ = FixedNumberPreConnector.__doc__

    def connect(self, projection):
        """Connect-up a Projection."""
        local             = numpy.ones(len(projection.pre), bool)
        weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
        delays_generator  = DelayGenerator(self.delays, local, self.safe)
        distance_matrix   = DistanceMatrix(projection.pre.positions, self.space)              
        self.progressbar(len(projection.post.local_cells))
        
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Warning: use of NativeRNG not implemented.")
            
        for count, tgt in enumerate(projection.post.local_cells.flat):
            # pick n neurons at random
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            
            candidates = projection.pre.all_cells.flatten()          
            if not self.allow_self_connections and projection.pre == projection.post:
                    i          = numpy.where(candidates == tgt)[0]
                    candidates = numpy.delete(candidates, i)
            sources = numpy.array([]) 
            while len(sources) < n: # if the number of requested cells is larger than the size of the
                                    # presynaptic population, we allow multiple connections for a given cell
                sources = numpy.concatenate((sources, projection.rng.permutation(candidates)[:n])) 
                            
            distance_matrix.set_source(tgt.position)
            sources = sources[:n]
            create  = projection.pre.id_to_index(sources).astype(int)
            weights = weights_generator.get(n, distance_matrix, create)
            delays  = delays_generator.get(n, distance_matrix, create)            
                 
            if len(sources) > 0:
                projection.connection_manager.convergent_connect(sources.tolist(), [tgt], weights, delays)
            
            self.progression(count)
