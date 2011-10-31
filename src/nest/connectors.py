"""
Connection method classes for nest

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""
from pyNN import random, core
from pyNN.connectors import Connector, AllToAllConnector, FixedProbabilityConnector, \
                            DistanceDependentProbabilityConnector, FixedNumberPreConnector, \
                            FixedNumberPostConnector, OneToOneConnector, SmallWorldConnector, \
                            FromListConnector, FromFileConnector, WeightGenerator, \
                            DelayGenerator, ProbaGenerator, DistanceMatrix, CSAConnector

import numpy
from pyNN.space import Space




class FastProbabilisticConnector(Connector):
    
    def __init__(self, projection, weights=0.0, delays=None, allow_self_connections=True, space=Space(), safe=True):

        Connector.__init__(self, weights, delays, space, safe)
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Use of NativeRNG not implemented.")
        else:
            self.rng = projection.rng
        if self.delays is None:
            self.delays = projection._simulator.state.min_delay
        self.N                 = projection.pre.size   
        idx                    = numpy.arange(self.N*rank(), self.N*(rank()+1))        
        self.M                 = num_processes()*self.N
        self.local             = numpy.ones(self.N, bool)        
        self.local_long        = numpy.zeros(self.M, bool)
        self.local_long[idx]   = True
        self.weights_generator = WeightGenerator(weights, self.local_long, projection, safe)
        self.delays_generator  = DelayGenerator(delays, self.local_long, kernel=projection._simulator.state, safe=safe)
        self.probas_generator  = ProbaGenerator(random.RandomDistribution('uniform',(0,1), rng=self.rng), self.local_long)
        self.distance_matrix   = DistanceMatrix(projection.pre.positions, self.space, self.local)
        self.projection        = projection
        self.candidates        = projection.pre.all_cells
        self.allow_self_connections = allow_self_connections
                
    def _probabilistic_connect(self, tgt, p, n_connections=None, rewiring=None):
        """
        Connect-up a Projection with connection probability p, where p may be either
        a float 0<=p<=1, or a dict containing a float array for each pre-synaptic
        cell, the array containing the connection probabilities for all the local
        targets of that pre-synaptic cell.
        """
        if numpy.isscalar(p) and p == 1:
            precreate = numpy.arange(self.N)
        else:
            rarr   = self.probas_generator.get(self.M)
            if not core.is_listlike(rarr) and numpy.isscalar(rarr): # if N=1, rarr will be a single number
                rarr = numpy.array([rarr])
            precreate = numpy.where(rarr < p)[0]  
        self.distance_matrix.set_source(tgt.position)
        
        if not self.allow_self_connections and self.projection.pre == self.projection.post:
            idx_tgt   = numpy.where(self.candidates == tgt)
            if len(idx_tgt) > 0:
                i     = numpy.where(precreate == idx_tgt[0])
                if len(i) > 0:
                    precreate = numpy.delete(precreate, i[0])
                
        if (rewiring is not None) and (rewiring > 0):
            idx = numpy.arange(0, self.N)          
            if not self.allow_self_connections and self.projection.pre == self.projection.post:
                i   = numpy.where(self.candidates == tgt)[0]
                idx = numpy.delete(idx, i)
            
            rarr    = self.probas_generator.get(self.M)[precreate]
            rewired = numpy.where(rarr < rewiring)[0]
            N       = len(rewired)
            if N > 0:
                new_idx            = (len(idx)-1) * self.probas_generator.get(self.M)[precreate]
                precreate[rewired] = idx[new_idx.astype(int)]    
        
        if (n_connections is not None) and (len(precreate) > 0):
            create = numpy.array([], int)
            while len(create) < n_connections: # if the number of requested cells is larger than the size of the
                                               # presynaptic population, we allow multiple connections for a given cell
                create = numpy.concatenate((create, self.projection.rng.permutation(precreate)))
            create = create[:n_connections]
        else:
            create = precreate   

        sources = self.candidates[create]        
        weights = self.weights_generator.get(self.M, self.distance_matrix, create)
        delays  = self.delays_generator.get(self.M, self.distance_matrix, create)        
        
        if len(sources) > 0:
            self.projection._convergent_connect(sources.tolist(), tgt, weights, delays)
    
    
class FastAllToAllConnector(AllToAllConnector):
    
    __doc__ = AllToAllConnector.__doc__
    
    def connect(self, projection):
        connector = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)        
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells):
            connector._probabilistic_connect(tgt, 1)
            self.progression(count)        
    

class FastFixedProbabilityConnector(FixedProbabilityConnector):
    
    __doc__ = FixedProbabilityConnector.__doc__    
        
    def connect(self, projection):
        connector = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells):
            connector._probabilistic_connect(tgt, self.p_connect)
            self.progression(count)
            

class FastDistanceDependentProbabilityConnector(DistanceDependentProbabilityConnector):
    
    __doc__ = DistanceDependentProbabilityConnector.__doc__
    
    def connect(self, projection):
        """Connect-up a Projection."""
        connector       = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        proba_generator = ProbaGenerator(self.d_expression, connector.local)
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells):
            connector.distance_matrix.set_source(tgt.position)
            proba  = proba_generator.get(connector.N, connector.distance_matrix)
            if proba.dtype == 'bool':
                proba = proba.astype(float)           
            connector._probabilistic_connect(tgt, proba, self.n_connections)
            self.progression(count)



class FixedNumberPreConnector(FixedNumberPreConnector):
    
    __doc__ = FixedNumberPreConnector.__doc__

    def connect(self, projection):
        connector = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells):
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            connector._probabilistic_connect(tgt, 1, n)
            self.progression(count)


class FastSmallWorldConnector(SmallWorldConnector):
    
    __doc__ = SmallWorldConnector.__doc__
    
    def connect(self, projection):
        """Connect-up a Projection."""
        connector       = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        proba_generator = ProbaGenerator(self.d_expression, connector.local)
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells):
            connector.distance_matrix.set_source(tgt.position)
            proba  = proba_generator.get(connector.N, connector.distance_matrix).astype(float)
            connector._probabilistic_connect(tgt, proba, self.n_connections, self.rewiring)
            self.progression(count)
