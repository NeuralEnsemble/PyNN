"""
Connection method classes for nest

$Id$
"""
from pyNN import random, common, core
from pyNN.connectors import *
import numpy
from pyNN.space import Space


class FastProbabilisticConnector(Connector):
    
    def __init__(self, projection, weights=0.0, delays=None, allow_self_connections=True, space=Space(), safe=True):

        Connector.__init__(self, weights, delays, space, safe)
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Use of NativeRNG not implemented.")
        else:
            self.rng = projection.rng
        
        self.local             = numpy.ones(len(projection.pre), bool)
        self.N                 = projection.pre.size
        self.weights_generator = WeightGenerator(weights, self.local, projection, safe)
        self.delays_generator  = DelayGenerator(delays, self.local, safe)
        self.probas_generator  = ProbaGenerator(RandomDistribution('uniform',(0,1), rng=self.rng), self.local)
        self.distance_matrix   = DistanceMatrix(projection.pre.positions, self.space, self.local)
        self.projection        = projection
        self.allow_self_connections = allow_self_connections
        
        
    def _probabilistic_connect(self, tgt, p):
        """
        Connect-up a Projection with connection probability p, where p may be either
        a float 0<=p<=1, or a dict containing a float array for each pre-synaptic
        cell, the array containing the connection probabilities for all the local
        targets of that pre-synaptic cell.
        """
        if numpy.isscalar(p) and p == 1:
            create = numpy.arange(self.local.sum())
        else:
            rarr   = self.probas_generator.get(self.N)
            if not core.is_listlike(rarr) and numpy.isscalar(rarr): # if N=1, rarr will be a single number
                rarr = numpy.array([rarr])
            create = numpy.where(rarr < p)[0]  
        self.distance_matrix.set_source(tgt.position)
        #create  = self.projection.pre.id_to_index(create).astype(int)
        sources = self.projection.pre.all_cells.flatten()[create]
        if not self.allow_self_connections and self.projection.pre == self.projection.post and tgt in sources:
            i       = numpy.where(sources == tgt)[0]
            sources = numpy.delete(sources, i)
            create  = numpy.delete(create, i)

        weights = self.weights_generator.get(self.N, self.distance_matrix, create)
        delays  = self.delays_generator.get(self.N, self.distance_matrix, create)        
        
        if len(sources) > 0:
            self.projection.connection_manager.convergent_connect(sources.tolist(), tgt, weights, delays)
    
    
class FastAllToAllConnector(AllToAllConnector):
    
    __doc__ = AllToAllConnector.__doc__
    
    def connect(self, projection):
        connector = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)        
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells.flat):
            connector._probabilistic_connect(tgt, 1)
            self.progression(count)        
    

class FastFixedProbabilityConnector(FixedProbabilityConnector):
    
    __doc__ = FixedProbabilityConnector.__doc__    
        
    def connect(self, projection):
        connector = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells.flat):
            connector._probabilistic_connect(tgt, self.p_connect)
            self.progression(count)
            

class FastDistanceDependentProbabilityConnector(DistanceDependentProbabilityConnector):
    
    __doc__ = DistanceDependentProbabilityConnector.__doc__
    
    def connect(self, projection):
        """Connect-up a Projection."""
        connector       = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        proba_generator = ProbaGenerator(self.d_expression, connector.local)
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells.flat):
            connector.distance_matrix.set_source(tgt.position)
            proba  = proba_generator.get(connector.N, connector.distance_matrix)
            if proba.dtype == 'bool':
                proba = proba.astype(float)
            connector._probabilistic_connect(tgt, proba)
            self.progression(count)



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
