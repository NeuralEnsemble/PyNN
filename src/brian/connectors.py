"""
Connection method classes for the brian module

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""
from pyNN.space import Space
from pyNN.random import RandomDistribution
import numpy
from pyNN import random, common, core
from pyNN.connectors import AllToAllConnector, \
                            ProbabilisticConnector, \
                            OneToOneConnector, \
                            FixedProbabilityConnector, \
                            DistanceDependentProbabilityConnector, \
                            FromListConnector, \
                            FromFileConnector, \
                            FixedNumberPreConnector, \
                            FixedNumberPostConnector, \
                            SmallWorldConnector, \
                            CSAConnector, \
                            WeightGenerator, \
                            DelayGenerator, \
                            ProbaGenerator

class FastProbabilisticConnector(ProbabilisticConnector):
    
    def __init__(self, projection, weights=0.0, delays=None, allow_self_connections=True, space=Space(), safe=True):
    
        ProbabilisticConnector.__init__(self, projection, weights, delays, allow_self_connections, space, safe)
        
    def _probabilistic_connect(self, src, p, n_connections=None):
        """
        Connect-up a Projection with connection probability p, where p may be either
        a float 0<=p<=1, or a dict containing a float array for each pre-synaptic
        cell, the array containing the connection probabilities for all the local
        targets of that pre-synaptic cell.
        """
        if numpy.isscalar(p) and p == 1:
            precreate = numpy.arange(self.size)
        else:
            rarr   = self.probas_generator.get(self.N)
            if not core.is_listlike(rarr) and numpy.isscalar(rarr): # if N=1, rarr will be a single number
                rarr = numpy.array([rarr])
            precreate = numpy.where(rarr < p)[0]  

        self.distance_matrix.set_source(src.position)        
        if not self.allow_self_connections and self.projection.pre == self.projection.post:
            idx_src   = numpy.where(self.candidates == src)
            if len(idx_src) > 0:
                i     = numpy.where(precreate == idx_src[0])
                if len(i) > 0:
                    precreate = numpy.delete(precreate, i[0])
                
        if (n_connections is not None) and (len(precreate) > 0):            
            create = numpy.array([], int)
            while len(create) < n_connections: # if the number of requested cells is larger than the size of the
                                               ## presynaptic population, we allow multiple connections for a given cell
                create = numpy.concatenate((create, self.projection.rng.permutation(precreate)))
            create = create[:n_connections]
        else:
            create = precreate            
        targets = self.candidates[create]        
        weights = self.weights_generator.get(self.N, self.distance_matrix, create)
        delays  = self.delays_generator.get(self.N, self.distance_matrix, create)        
        
        homogeneous = numpy.isscalar(self.delays_generator.source)
        if len(targets) > 0:
            self.projection._divergent_connect(src, targets.tolist(), weights, delays, homogeneous)


class FastAllToAllConnector(AllToAllConnector):
    
    __doc__ = AllToAllConnector.__doc__
    
    def connect(self, projection):
        connector = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)        
        self.progressbar(len(projection.pre.local_cells))
        for count, tgt in enumerate(projection.pre.local_cells):
            connector._probabilistic_connect(tgt, 1)
            self.progression(count)        
    

class FastFixedProbabilityConnector(FixedProbabilityConnector):
    
    __doc__ = FixedProbabilityConnector.__doc__    
        
    def connect(self, projection):
        connector = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        self.progressbar(len(projection.pre.local_cells))
        for count, tgt in enumerate(projection.pre.local_cells):
            connector._probabilistic_connect(tgt, self.p_connect)
            self.progression(count)
            

class FastDistanceDependentProbabilityConnector(DistanceDependentProbabilityConnector):
    
    __doc__ = DistanceDependentProbabilityConnector.__doc__
    
    def connect(self, projection):
        """Connect-up a Projection."""
        connector       = FastProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        proba_generator = ProbaGenerator(self.d_expression, connector.local)
        self.progressbar(len(projection.pre.local_cells))
        for count, tgt in enumerate(projection.pre.local_cells):
            connector.distance_matrix.set_source(tgt.position)
            proba  = proba_generator.get(connector.N, connector.distance_matrix)
            if proba.dtype == 'bool':
                proba = proba.astype(float)           
            connector._probabilistic_connect(tgt, proba, self.n_connections)
            self.progression(count)

class FastOneToOneConnector(OneToOneConnector):
    
    __doc__ = OneToOneConnector.__doc__
    
    def connect(self, projection):
        """Connect-up a Projection."""        
        if projection.pre.size == projection.post.size:
            N                 = projection.post.size
            local             = projection.post._mask_local
            if isinstance(self.weights, basestring) or isinstance(self.delays, basestring):
                raise Exception('Expression for weights or delays is not supported for OneToOneConnector !')
            weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
            delays_generator  = DelayGenerator(self.delays, local, self.safe)                
            weights           = weights_generator.get(N)
            delays            = delays_generator.get(N)
            self.progressbar(len(projection.post.local_cells))                        
            count             = 0            
            create            = numpy.arange(0, N)[local]
            sources           = projection.pre.all_cells[create] 
            homogeneous       = numpy.isscalar(delays_generator.source)
            
            for tgt, src, w, d in zip(projection.post.local_cells, sources, weights, delays):
                # the float is in case the values are of type numpy.float64, which NEST chokes on
                projection._divergent_connect(src, [tgt], float(w), float(d), homogeneous)
                self.progression(count)
                count += 1
        else:
            raise errors.InvalidDimensionsError("OneToOneConnector does not support presynaptic and postsynaptic Populations of different sizes.")
