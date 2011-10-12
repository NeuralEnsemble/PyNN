import numpy, logging, sys
from pyNN import errors, common, core, random, utility
from pyNN.space import Space
from pyNN.random import RandomDistribution
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                    fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                    sin, sinh, sqrt, tan, tanh, maximum, minimum

logger = logging.getLogger("PyNN")

class ConnectionAttributeGenerator(object):
    
    def __init__(self, source, local_mask, safe=True):
        self.source     = source
        self.local_mask = local_mask
        self.local_size = local_mask.sum()
        self.safe       = safe
        if self.safe:
            self.get = self.get_safe    
        if isinstance(self.source, numpy.ndarray):
            self.source_iterator = iter(self.source)
    
    def check(self, data):
        return data
        
    def extract(self, N, distance_matrix=None, sub_mask=None):
        #local_mask is supposed to be a mask of booleans, while 
        #sub_mask is a list of cells ids.
        if isinstance(self.source, basestring):
            assert distance_matrix is not None
            d      = distance_matrix.as_array(sub_mask)
            values = eval(self.source)
            return values
        elif hasattr(self.source, 'func_name'):
            assert distance_matrix is not None
            d      = distance_matrix.as_array(sub_mask)
            values = self.source(d)
            return values
        elif numpy.isscalar(self.source):
            if sub_mask is None:
                values = numpy.ones((self.local_size,))*self.source
            else:
                values = numpy.ones((len(sub_mask),))*self.source
            return values
        elif isinstance(self.source, RandomDistribution):
            if sub_mask is None:
                values = self.source.next(N, mask_local=self.local_mask)
            else:
                values = self.source.next(len(sub_mask), mask_local=self.local_mask)
            return values
        elif isinstance(self.source, numpy.ndarray):
            source_row = self.source_iterator.next()
            values     = source_row[self.local_mask]
            if sub_mask is not None:
                values = values[sub_mask]
            return values    

    def get_safe(self, N, distance_matrix=None, sub_mask=None):
        return self.check(self.extract(N, distance_matrix, sub_mask))
    
    def get(self, N, distance_matrix=None, sub_mask=None):
        return self.extract(N, distance_matrix, sub_mask)


class WeightGenerator(ConnectionAttributeGenerator):
    
    def __init__(self, source, local_mask, projection, safe=True):
      ConnectionAttributeGenerator.__init__(self, source, local_mask, safe)
      self.projection     = projection
      self.is_conductance = common.is_conductance(projection.post.all_cells[0])
      
    def check(self, weight):
        if weight is None:
            weight = DEFAULT_WEIGHT
        if core.is_listlike(weight):
            weight     = numpy.array(weight)
            nan_filter = (1-numpy.isnan(weight)).astype(bool) # weight arrays may contain NaN, which should be ignored
            filtered_weight = weight[nan_filter]
            all_negative = (filtered_weight<=0).all()
            all_positive = (filtered_weight>=0).all()
            if not (all_negative or all_positive):
                raise errors.InvalidWeightError("Weights must be either all positive or all negative")
        elif numpy.isscalar(weight):
            all_positive = weight >= 0
            all_negative = weight < 0
        else:
            raise Exception("Weight must be a number or a list/array of numbers.")
        if self.is_conductance or self.projection.synapse_type == 'excitatory':
            if not all_positive:
                raise errors.InvalidWeightError("Weights must be positive for conductance-based and/or excitatory synapses")
        elif self.is_conductance==False and self.projection.synapse_type == 'inhibitory':
            if not all_negative:
                raise errors.InvalidWeightError("Weights must be negative for current-based, inhibitory synapses")
        else: # is_conductance is None. This happens if the cell does not exist on the current node.
            logger.debug("Can't check weight, conductance status unknown.")
        return weight
        
    

class DelayGenerator(ConnectionAttributeGenerator):

    def __init__(self, source, local_mask, safe=True):
        ConnectionAttributeGenerator.__init__(self, source, local_mask, safe)
        self.min_delay = common.get_min_delay()
        self.max_delay = common.get_max_delay()
        
    def check(self, delay):
        all_negative = (delay<=self.max_delay).all()
        all_positive = (delay>=self.min_delay).all()# If the delay is too small , we have to throw an error
        if not (all_negative and all_positive):
            raise errors.ConnectionError("delay (%s) is out of range [%s,%s]" % (delay, common.get_min_delay(), common.get_max_delay()))
        return delay    

class ProbaGenerator(ConnectionAttributeGenerator):
    pass


class DistanceMatrix(object):
    
    def __init__(self, B, space, mask=None):
        assert B.shape[0] == 3
        self.space = space
        if mask is not None:
            self.B = B[:,mask]
        else:
            self.B = B
        
    def as_array(self, sub_mask=None):
        if self._distance_matrix is None and self.A is not None:
            if sub_mask is None:
                self._distance_matrix = self.space.distances(self.A, self.B)[0]
            else:
                self._distance_matrix = self.space.distances(self.A, self.B[:,sub_mask])[0]
        return self._distance_matrix
        
    def set_source(self, A):
        assert A.shape == (3,)
        self.A = A
        self._distance_matrix = None        


class Connector(object):
    
    def __init__(self, weights=0.0, delays=None, space=Space(), safe=True, verbose=False):
        self.weights = weights
        self.space   = space
        self.safe    = safe
        self.verbose = verbose
        min_delay    = common.get_min_delay()
        if delays is None:
            self.delays = min_delay
        else:
            if core.is_listlike(delays):
                assert min(delays) >= min_delay
            elif not (isinstance(delays, basestring) or isinstance(delays, RandomDistribution)):
                assert delays >= min_delay
            self.delays = delays        
        
    def connect(self, projection):
        pass
    
    def progressbar(self, N):
        self.prog = utility.ProgressBar(0, N, 20, mode='fixed')        
    
    def progression(self, count):
        self.prog.update_amount(count)
        if self.verbose and common.rank() == 0:           
            print self.prog, "\r",
            sys.stdout.flush()
            


class ProbabilisticConnector(Connector):
    
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
            self.projection._convergent_connect(sources.tolist(), tgt, weights, delays)
    
    
class AllToAllConnector(Connector):
    """
    Connects all cells in the presynaptic population to all cells in the
    postsynaptic population.
    """
    
    def __init__(self, allow_self_connections=True, weights=0.0, delays=None, space=Space(), safe=True, verbose=False):
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
        Connector.__init__(self, weights, delays, space, safe, verbose)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        
    def connect(self, projection):
        connector = ProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)        
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells.flat):
            connector._probabilistic_connect(tgt, 1)
            self.progression(count)        
    

class FixedProbabilityConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability is constant.
    """
    
    def __init__(self, p_connect, allow_self_connections=True, weights=0.0, delays=None, space=Space(), 
                       safe=True, verbose=False):
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
        Connector.__init__(self, weights, delays, space, safe, verbose)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        self.p_connect = float(p_connect)
        assert 0 <= self.p_connect
        
    def connect(self, projection):
        connector = ProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells.flat):
            connector._probabilistic_connect(tgt, self.p_connect)
            self.progression(count)
            

class DistanceDependentProbabilityConnector(ProbabilisticConnector):
    """
    For each pair of pre-post cells, the connection probability depends on distance.
    """
    
    def __init__(self, d_expression, allow_self_connections=True,
                 weights=0.0, delays=None, space=Space(), safe=True, verbose=False):
        """
        Create a new connector.
        
        `d_expression` -- the right-hand side of a valid python expression for
            probability, involving 'd', e.g. "exp(-abs(d))", or "d<3"
        `space` -- a Space object.
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created, or a DistanceDependence object. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        Connector.__init__(self, weights, delays, space, safe, verbose)
        assert isinstance(d_expression, str)
        try:
            d = 0; assert 0 <= eval(d_expression), eval(d_expression)
            d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
        except ZeroDivisionError, err:
            raise ZeroDivisionError("Error in the distance expression %s. %s" % (d_expression, err))
        self.d_expression = d_expression
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        
    def connect(self, projection):
        """Connect-up a Projection."""
        connector       = ProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        proba_generator = ProbaGenerator(self.d_expression, connector.local)
        self.progressbar(len(projection.post.local_cells))
        for count, tgt in enumerate(projection.post.local_cells.flat):
            connector.distance_matrix.set_source(tgt.position)
            proba  = proba_generator.get(connector.N, connector.distance_matrix)
            if proba.dtype == 'bool':
                proba = proba.astype(float)
            connector._probabilistic_connect(tgt, proba)
            self.progression(count)
    

class FromListConnector(Connector):
    """
    Make connections according to a list.
    """
    
    def __init__(self, conn_list, safe=True, verbose=False):
        """
        Create a new connector.
        
        `conn_list` -- a list of tuples, one tuple for each connection. Each
                       tuple should contain:
                          (pre_addr, post_addr, weight, delay)
                       where pre_addr is the address (a tuple) of the presynaptic
                       neuron, and post_addr is the address of the postsynaptic
                       neuron.
        """
        # needs extending for dynamic synapses.
        Connector.__init__(self, 0., common.get_min_delay(), safe=safe, verbose=verbose)
        self.conn_list = conn_list        
        
    def connect(self, projection):
        """Connect-up a Projection."""
        # slow: should maybe sort by pre
        self.progressbar(len(self.conn_list))
        for count, i in enumerate(xrange(len(self.conn_list))):
            src, tgt, weight, delay = self.conn_list[i][:]
            src = projection.pre[tuple(src)]           
            tgt = projection.post[tuple(tgt)]
            projection._divergent_connect(src, [tgt], weight, delay)
            self.progression(count)
            

class FromFileConnector(FromListConnector):
    """
    Make connections according to a list read from a file.
    """
    
    def __init__(self, filename, distributed=False, safe=True, verbose=False):
        """
        Create a new connector.
        
        `filename` -- name of a text file containing a list of connections, in
                      the format required by `FromListConnector`.
        `distributed` -- if this is True, then each node will read connections
                         from a file called `filename.x`, where `x` is the MPI
                         rank. This speeds up loading connections for
                         distributed simulations.
        """
        Connector.__init__(self, 0., common.get_min_delay(), safe=safe, verbose=verbose)
        self.filename = filename
        self.distributed = distributed

    def connect(self, projection):
        """Connect-up a Projection."""
        if self.distributed:
            self.filename += ".%d" % common.rank()
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
        self.conn_list = input_tuples
        FromListConnector.connect(self, projection)



class FixedNumberPostConnector(Connector):
    """
    Each pre-synaptic neuron is connected to exactly n post-synaptic neurons
    chosen at random.
    
    If n is less than the size of the post-synaptic population, there are no
    multiple connections, i.e., no instances of the same pair of neurons being
    multiply connected. If n is greater than the size of the post-synaptic
    population, all possible single connections are made before starting to add
    duplicate connections.
    """
    
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=None, space=Space(), safe=True, verbose=False):
        """
        Create a new connector.
        
        `n` -- either a positive integer, or a `RandomDistribution` that produces
               positive integers. If `n` is a `RandomDistribution`, then the
               number of post-synaptic neurons is drawn from this distribution
               for each pre-synaptic neuron.
        `allow_self_connections` -- if the connector is used to connect a
               Population to itself, this flag determines whether a neuron is
               allowed to connect to itself, or only to other neurons in the
               Population.
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        Connector.__init__(self, weights, delays, space, safe, verbose)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        if isinstance(n, int):
            self.n = n
            assert n >= 0
        elif isinstance(n, random.RandomDistribution):
            self.rand_distr = n
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) >= 0), "the random distribution produces negative numbers"
        else:
            raise Exception("n must be an integer or a RandomDistribution object")
        
    def connect(self, projection):
        """Connect-up a Projection."""        
        local             = numpy.ones(len(projection.post), bool)
        weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
        delays_generator  = DelayGenerator(self.delays, local, self.safe)
        distance_matrix   = DistanceMatrix(projection.post.positions, self.space)
        self.progressbar(len(projection.pre))
        
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Use of NativeRNG not implemented.")        
        
        for count, src in enumerate(projection.pre.all()):
            # pick n neurons at random
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            
            candidates  = projection.post.all_cells.flatten()
            if not self.allow_self_connections and projection.pre == projection.post:                
                idx        = numpy.where(candidates == src)[0]
                candidates = numpy.delete(candidates, idx)

            targets = numpy.array([])          
            while len(targets) < n: # if the number of requested cells is larger than the size of the
                                    # postsynaptic population, we allow multiple connections for a given cell   
                targets = numpy.concatenate((targets, projection.rng.permutation(candidates)[:n]))                 
            
            distance_matrix.set_source(src.position)
            targets = targets[:n]
            create  = projection.post.id_to_index(targets).astype(int)
            weights = weights_generator.get(n, distance_matrix, create)
            delays  = delays_generator.get(n, distance_matrix, create)      
               
            if len(targets) > 0:
                projection._divergent_connect(src, targets.tolist(), weights, delays)
            
            self.progression(count)
            

class FixedNumberPreConnector(Connector):
    """
    Each post-synaptic neuron is connected to exactly n pre-synaptic neurons
    chosen at random.
    
    If n is less than the size of the pre-synaptic population, there are no
    multiple connections, i.e., no instances of the same pair of neurons being
    multiply connected. If n is greater than the size of the pre-synaptic
    population, all possible single connections are made before starting to add
    duplicate connections.
    """
    
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=None, space=Space(), safe=True, verbose=False):
        """
        Create a new connector.
        
        `n` -- either a positive integer, or a `RandomDistribution` that produces
               positive integers. If `n` is a `RandomDistribution`, then the
               number of pre-synaptic neurons is drawn from this distribution
               for each post-synaptic neuron.
        `allow_self_connections` -- if the connector is used to connect a
            Population to itself, this flag determines whether a neuron is
            allowed to connect to itself, or only to other neurons in the
            Population.
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        Connector.__init__(self, weights, delays, space, safe, verbose)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        if isinstance(n, int):
            self.n = n
            assert n >= 0
        elif isinstance(n, random.RandomDistribution):
            self.rand_distr = n
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) >= 0), "the random distribution produces negative numbers"
        else:
            raise Exception("n must be an integer or a RandomDistribution object")

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
                projection._convergent_connect(sources, tgt, weights, delays)
            self.progression(count)
            

class OneToOneConnector(Connector):
    """
    Where the pre- and postsynaptic populations have the same size, connect
    cell i in the presynaptic population to cell i in the postsynaptic
    population for all i.
    """
    #In fact, despite the name, this should probably be generalised to the
    #case where the pre and post populations have different dimensions, e.g.,
    #cell i in a 1D pre population of size n should connect to all cells
    #in row i of a 2D post population of size (n,m).
    
    
    def __init__(self, weights=0.0, delays=None, space=Space(), safe=True, verbose=False):
        """
        Create a new connector.
        
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        Connector.__init__(self, weights, delays, space, verbose)
        self.space = space
        self.safe  = safe
    
    def connect(self, projection):
        """Connect-up a Projection."""        
        if projection.pre.dim == projection.post.dim:
            N                 = projection.post.size
            local             = projection.post._mask_local.flatten()
            if isinstance(self.weights, basestring) or isinstance(self.delays, basestring):
                raise Exception('Expression for weights or delays is not supported for OneToOneConnector !')
            weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
            delays_generator  = DelayGenerator(self.delays, local, self.safe)                
            weights           = weights_generator.get(N)
            delays            = delays_generator.get(N)
            self.progressbar(len(projection.post.local_cells))                        
            count             = 0
            
            for tgt, w, d in zip(projection.post.local_cells, weights, delays):
                src = projection.pre.index(projection.post.id_to_index(tgt))
                
                # the float is in case the values are of type numpy.float64, which NEST chokes on
                projection._divergent_connect(src, [tgt], float(w), float(d))
                self.progression(count)
                count += 1
        else:
            raise errors.InvalidDimensionsError("OneToOneConnector does not support presynaptic and postsynaptic Populations of different sizes.")


class SmallWorldConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability depends on distance.
    """
    
    def __init__(self, degree, rewiring, allow_self_connections=True,
                 weights=0.0, delays=None, space=Space(), safe=True, verbose=False):
        """
        Create a new connector.
        
        `degree` -- the region lenght where nodes will be connected locally
        `rewiring` -- the probability of rewiring each eadges 
        `space` -- a Space object.
        `allow_self_connections` -- if the connector is used to connect a
            Population to itself, this flag determines whether a neuron is
            allowed to connect to itself, or only to other neurons in the
            Population.        
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created, or a DistanceDependence object. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        Connector.__init__(self, weights, delays, space, safe, verbose)
        assert 0 <= rewiring <= 1
        assert isinstance(allow_self_connections, bool)        
        self.rewiring               = rewiring    
        self.d_expression           = "d < %g" %degree        
        self.allow_self_connections = allow_self_connections
        
    def _smallworld_connect(self, src, p):
        """
        Connect-up a Projection with connection probability p, where p may be either
        a float 0<=p<=1, or a dict containing a float array for each pre-synaptic
        cell, the array containing the connection probabilities for all the local
        targets of that pre-synaptic cell.
        """
        rarr   = self.probas_generator.get(self.N)
        if not core.is_listlike(rarr) and numpy.isscalar(rarr): # if N=1, rarr will be a single number
            rarr = numpy.array([rarr])
        create = numpy.where(rarr < p)[0]  
        self.distance_matrix.set_source(src.position)        
        
        targets    = self.candidates[create]        
        candidates = self.projection.post.all_cells.flatten()          
        if not self.allow_self_connections and projection.pre == projection.post:
            i          = numpy.where(candidates == src)[0]
            candidates = numpy.delete(candidates, i)
        
        rarr            = self.probas_generator.get(len(create))
        rewired         = rarr < self.rewiring
        if sum(rewired) > 0:
            idx              = numpy.random.random_integers(0, len(candidates)-1, sum(rewired))
            targets[rewired] = candidates[idx]
        create          = self.projection.post.id_to_index(targets).astype(int)
        weights         = self.weights_generator.get(self.N, self.distance_matrix, create)
        delays          = self.delays_generator.get(self.N, self.distance_matrix, create)      
                    
        if len(targets) > 0:
            self.projection._divergent_connect(src, targets.tolist(), weights, delays)
    
    def connect(self, projection):
        """Connect-up a Projection."""
        local                  = numpy.ones(len(projection.post), bool)
        self.N                 = projection.post.size        
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Use of NativeRNG not implemented.")
        else:
            self.rng = projection.rng
        self.weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
        self.delays_generator  = DelayGenerator(self.delays, local, self.safe)
        self.probas_generator  = ProbaGenerator(RandomDistribution('uniform',(0,1), rng=self.rng), local)
        self.distance_matrix   = DistanceMatrix(projection.post.positions, self.space, local)
        self.projection        = projection
        self.candidates        = self.projection.post.all_cells.flatten()
        self.progressbar(len(projection.pre))        
        proba_generator = ProbaGenerator(self.d_expression, local)
        for count, src in enumerate(projection.pre.all()):     
            self.distance_matrix.set_source(src.position)
            proba = proba_generator.get(self.N, self.distance_matrix).astype(float)
            self._smallworld_connect(src, proba)
            self.progression(count)   