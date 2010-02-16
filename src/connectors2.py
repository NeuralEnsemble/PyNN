import numpy
from pyNN import common
from pyNN.space import Space
from pyNN.random import RandomDistribution
from numpy import exp

class ConnectionAttributeGenerator(object):
    
    def __init__(self, source, local_mask):
        self.source = source
        self.local_mask = local_mask
        if isinstance(self.source, numpy.ndarray):
            self.source_iterator = iter(self.source)
    
    def get(self, connectivity_matrix, distance_matrix):
        local_value_mask = self.local_mask[connectivity_matrix]
        if common.is_number(self.source):
            return numpy.ones((local_value_mask.sum(),))*self.source
        elif isinstance(self.source, RandomDistribution):
            all_values = self.source.next(connectivity_matrix.sum(),
                                          mask_local=False)
            local_values = all_values[local_value_mask]
            return local_values
        elif isinstance(self.source, numpy.ndarray):
            source_row = self.source_iterator.next()
            assert source_row.shape == connectivity_matrix.shape
            return source_row[local_value_mask]
        elif isinstance(self.source, basestring):
            d = distance_matrix.as_array(mask=local_value_mask)
            values = eval(self.source)
            return values


class WeightGenerator(ConnectionAttributeGenerator):
    pass

class DelayGenerator(ConnectionAttributeGenerator):
    pass


class DistanceMatrix(object):
    
    def __init__(self, A, B, space):
        assert A.shape == (3,)
        assert B.shape[0] == 3
        self.space = space
        self.A = A
        self.B = B
        self._distance_matrix = None
        self._last_mask = None
        
    def as_array(self, mask=None):
        if self._distance_matrix is None or mask != self._last_mask:
            if mask is not None and mask.sum() < mask.size:
                local_B = ((self.B.T)[mask]).T
            else:
                local_B = self.B
            self._distance_matrix = self.space.distances(self.A, local_B)[0]
            self._last_mask = mask
        return self._distance_matrix
        

class Connector(object):
    
    def __init__(self, weights=0.0, delays=None):
        self.weights = weights
        min_delay = common.get_min_delay()
        if delays is None:
            self.delays = min_delay
        else:
            if common.is_listlike(delays):
                assert min(delays) >= min_delay
            else:
                assert delays >= min_delay
            self.delays = delays
    
    def connect(self, projection):
        pass
    
    
    
    
class AllToAllConnector(Connector):
    """
    Connects all cells in the presynaptic population to all cells in the
    postsynaptic population.
    """
    
    def __init__(self, allow_self_connections=True,
                 weights=0.0, delays=None, space=Space()):
        """
        Create a new connector.
        
        `allow_self_connections` -- if the connector is used to connect a
            Population to itself, this flag determines whether a neuron is
            allowed to connect to itself, or only to other neurons in the
            Population.
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        `space` -- a `Space` object, needed if you wish to specify distance-
                   dependent weights or delays
        """
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        self.space = space
        
        
    def connect(self, projection):       
        local = projection.post._mask_local.flatten()
        weight_generator = WeightGenerator(self.weights, local)
        delay_generator = DelayGenerator(self.delays, local)
        
        global_target_mask = numpy.ones((projection.post.size,), dtype=bool)
        targets = projection.post.local_cells.tolist()
        if len(targets) > 0:
            for src in projection.pre.all():
                distance_matrix = DistanceMatrix(src.position,
                                                 projection.post.positions,
                                                 space=self.space)
                weights = weight_generator.get(global_target_mask, distance_matrix)
                delays = delay_generator.get(global_target_mask, distance_matrix)
                projection.connection_manager.connect(src, targets, weights, delays)

    

class FixedProbabilityConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability is constant.
    """
    
    def __init__(self, p_connect, allow_self_connections=True,
                 weights=0.0, delays=None, space=Space()):
        """
        Create a new connector.
        
        `p_connect` -- a float between zero and one. Each potential connection
                       is created with this probability.
        `allow_self_connections` -- if the connector is used to connect a
            Population to itself, this flag determines whether a neuron is
            allowed to connect to itself, or only to other neurons in the
            Population.
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        `space` -- a `Space` object, needed if you wish to specify distance-
                   dependent weights or delays
        """
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        self.p_connect = float(p_connect)
        self.space = space
        assert 0 <= self.p_connect
        
    def connect(self, projection):
        assert projection.rng.parallel_safe
        p = self.p_connect
        local = projection.post._mask_local.flatten()
        weight_generator = WeightGenerator(self.weights, local)
        delay_generator = DelayGenerator(self.delays, local)
        for src in projection.pre.all():
            
            N = projection.post.size
            rarr = projection.rng.next(N, 'uniform', (0,1), mask_local=False)
            if not common.is_listlike(rarr) and common.is_number(rarr): # if N=1, rarr will be a single number
                rarr = numpy.array([rarr])
            global_target_mask = rarr < p
            local_target_mask = global_target_mask[local]
            targets = projection.post.local_cells[local_target_mask].tolist()
            
            if len(targets) > 0:
                distance_matrix = DistanceMatrix(src.position,
                                                 projection.post.positions,
                                                 space=self.space)
                weights = weight_generator.get(global_target_mask, distance_matrix)
                delays = delay_generator.get(global_target_mask, distance_matrix)
                projection.connection_manager.connect(src, targets, weights, delays)

