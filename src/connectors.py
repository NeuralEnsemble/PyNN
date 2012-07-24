"""
Defines a common implementation of the built-in PyNN Connector classes.

Simulator modules may use these directly, or may implement their own versions
for improved performance.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy, logging, sys, re
from pyNN import errors, common, core, random, utility, recording, descriptions
from pyNN.space import Space
from pyNN.recording import files
from pyNN.random import RandomDistribution
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                    fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                    sin, sinh, sqrt, tan, tanh, maximum, minimum
try:
    import csa
    haveCSA = True
except ImportError:
    haveCSA = False

logger = logging.getLogger("PyNN")

DEFAULT_WEIGHT = 0.0

def expand_distances(d_expression):
    """
    Check if a distance expression contains at least one term d[x]. If yes, then
    the distances are expanded and we assume the user has specified an
    expression such as d[0] + d[2].
    """
    regexpr = re.compile(r'.*d\[\d*\].*')
    if regexpr.match(d_expression):
        return True
    return False
            

class ConnectionAttributeGenerator(object):
    """
    Connection attributes, such as weights and delays, may be specified as:
        - a single numerical value, in which case all connections have this value
        - a numpy array of the same size as the number of connections
        - a RandomDistribution object
        - a function of the distance between the source and target cells
        
    This class encapsulates all these different possibilities in order to
    present a uniform interface.
    """
    
    def __init__(self, source, local_mask, safe=True):
        """
        Create a new %s.
        
        source - something that may be used to obtain connection attribute values
        local_mask - a boolean array indicating which of the post-synaptic cells
                     are on the local machine
        safe - whether to check that values are within the appropriate range. These
               checks can be slow, so safe=False allows you to turn them off once
               you're certain your code is working correctly.
        """ % self.__class__.__name__
        self.source     = source
        self.local_mask = local_mask
        self.safe       = safe
        if self.safe:
            self.get = self.get_safe
        if isinstance(self.source, list):
            self.source = numpy.array(self.source, dtype=numpy.int)
        if isinstance(self.source, numpy.ndarray):
            self.source_iterator = iter(self.source)
    
    def check(self, data):
        """
        This method should be over-ridden by sub-classes.
        """
        return data
        
    def extract(self, N, distance_matrix=None, sub_mask=None):
        """
        Return an array of values for a connection attribute.
        
        N - number of values to be returned over the entire simulation. If
            running a distributed simulation, the number returned on any given
            node will be smaller.
        distance_matrix - a DistanceMatrix object, used for calculating
                          distance-dependent attributes.
        sub-mask - a sublist of the ids we want compute some values with. For
                   example in parallel, distances shoudl be computed only between a source
                   and local targets, since only connections with those targets are established. 
                   Avoid useless computations...                 
                   
        """
        if isinstance(self.source, basestring):
            assert distance_matrix is not None            
            if expand_distances(self.source):            
                d = distance_matrix.as_array(sub_mask, expand=True)
            else:
                d = distance_matrix.as_array(sub_mask)
            values = eval(self.source)
            return values
        elif callable(self.source):
            assert distance_matrix is not None
            d      = distance_matrix.as_array(sub_mask)
            values = self.source(d)
            return values
        elif numpy.isscalar(self.source):
            if sub_mask is None:
                values = numpy.ones((self.local_mask.sum(),))*self.source
            else:
                values = numpy.ones((len(sub_mask),))*self.source
            return values # seems a bit wasteful to return an array of identical values
        elif isinstance(self.source, RandomDistribution):
            if sub_mask is None:
                values = self.source.next(N, mask_local=self.local_mask)
            else:
                data  = self.source.next(N, mask_local=self.local_mask)
                if type(data) == numpy.float:
                    data = numpy.array([data])
                values = data[sub_mask]
            return values
        elif isinstance(self.source, numpy.ndarray):
            if len(self.source.shape) == 1: # for OneToOneConnector or AllToAllConnector used from or to only one Neuron   
                ## First we reshape the data, and reinit the source_iterator. This is a patch since data are selected according
                ## to the post_synaptic index with local_mask. So therefore, the weights should be in the form of a matrix of size
                ## n_pre lines and n_post columns. If this is a vector, we need to transpose it. 
                self.source = self.source.reshape((len(self.source), 1)) 
                self.source_iterator = iter(self.source)
            if len(self.source.shape) == 2:
                source_row = self.source_iterator.next()
                values     = source_row[self.local_mask]
            else:
                raise Exception()
            if sub_mask is not None:
                values = values[sub_mask]
            return values
        else:
            raise Exception("Invalid source (%s)" % type(self.source))

    def get_safe(self, N, distance_matrix=None, sub_mask=None):
        return self.check(self.extract(N, distance_matrix, sub_mask))
    
    def get(self, N, distance_matrix=None, sub_mask=None):
        return self.extract(N, distance_matrix, sub_mask)


class WeightGenerator(ConnectionAttributeGenerator):
    """Generator for synaptic weights. %s""" % ConnectionAttributeGenerator.__doc__
    
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
    """Generator for synaptic delays. %s""" % ConnectionAttributeGenerator.__doc__

    def __init__(self, source, local_mask, kernel, safe=True):
        ConnectionAttributeGenerator.__init__(self, source, local_mask, safe)
        assert hasattr(kernel, "min_delay")
        self.kernel = kernel
        
    def check(self, delay):
        min_delay = self.kernel.min_delay
        max_delay = self.kernel.max_delay
        all_negative = (delay<=max_delay).all()
        all_positive = (delay>=min_delay).all()# If the delay is too small , we have to throw an error
        if not (all_negative and all_positive):
            raise errors.ConnectionError("delay (%s) is out of range [%s,%s]" % (delay, min_delay, max_delay))
        return delay


class ProbaGenerator(ConnectionAttributeGenerator):
    pass


class DistanceMatrix(object):
    # should probably move to space module
    
    def __init__(self, B, space, mask=None):
        assert B.shape[0] == 3, B.shape
        self.space = space
        if mask is not None:
            self.B = B[:,mask]
        else:
            self.B = B
        
    def as_array(self, sub_mask=None, expand=False):
        if self._distance_matrix is None and self.A is not None:
            if sub_mask is None:
                self._distance_matrix = self.space.distances(self.A, self.B, expand)
            else:
                self._distance_matrix = self.space.distances(self.A, self.B[:,sub_mask], expand)
            if expand:
                N = self._distance_matrix.shape[2]
                self._distance_matrix = self._distance_matrix.reshape((3, N))
            else:
                self._distance_matrix = self._distance_matrix[0]
        return self._distance_matrix
        
    def set_source(self, A):
        assert A.shape == (3,), A.shape
        self.A = A
        self._distance_matrix = None        


class Connector(object):
    
    def __init__(self, weights=0.0, delays=None, space=Space(), safe=True, verbose=False):
        self.weights = weights
        self.space   = space
        self.safe    = safe
        self.verbose = verbose
        self.delays = delays
    
    def connect(self, projection):
        raise NotImplementedError()
    
    def progressbar(self, N):
        self.prog = utility.ProgressBar(0, N, 20, mode='fixed')        
    
    def progression(self, count, mpi_rank):
        self.prog.update_amount(count)
        if self.verbose and mpi_rank == 0:           
            print self.prog, "\r",
            sys.stdout.flush()
    
    def get_parameters(self):
        P = {}
        for name in self.parameter_names:
            P[name] = getattr(self, name)
        return P
    
    def describe(self, template='connector_default.txt', engine='default'):
        """
        Returns a human-readable description of the connection method.
        
        The output may be customized by specifying a different template
        togther with an associated template engine (see ``pyNN.descriptions``).
        
        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {'name': self.__class__.__name__,
                   'parameters': self.get_parameters(),
                   'weights': self.weights,
                   'delays': self.delays}
        return descriptions.render(engine, template, context)


class ProbabilisticConnector(Connector):
    
    def __init__(self, projection, weights=0.0, delays=None,
                 allow_self_connections=True, space=Space(), safe=True):

        Connector.__init__(self, weights, delays, space, safe)
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Use of NativeRNG not implemented.")
        else:
            self.rng = projection.rng
        if self.delays is None:
            self.delays = projection._simulator.state.min_delay
        self.local             = projection.post._mask_local
        self.N                 = projection.post.size
        self.weights_generator = WeightGenerator(weights, self.local, projection, safe)
        self.delays_generator  = DelayGenerator(self.delays, self.local, kernel=projection._simulator.state, safe=safe)
        self.probas_generator  = ProbaGenerator(RandomDistribution('uniform', (0,1), rng=self.rng), self.local)
        self._distance_matrix  = None
        self.projection        = projection
        self.candidates        = projection.post.local_cells
        self.size              = self.local.sum()
        self.allow_self_connections = allow_self_connections
       
    @property 
    def distance_matrix(self):
        """
        We want to avoid calculating positions if it is not necessary, so we
        delay it until the distance matrix is actually used.
        """
        if self._distance_matrix is None:
            self._distance_matrix = DistanceMatrix(self.projection.post.positions, self.space, self.local)
        return self._distance_matrix
        
    def _probabilistic_connect(self, src, p, n_connections=None):
        """
        Connect-up a Projection with connection probability p, where p may be either
        a float 0<=p<=1, or a dict containing a float array for each pre-synaptic
        cell, the array containing the connection probabilities for all the local
        targets of that pre-synaptic cell.
        """
        if numpy.isscalar(p) and p == 1:
            precreate = numpy.arange(self.size, dtype=numpy.int)
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
            create = numpy.array([], dtype=numpy.int)
            while len(create) < n_connections: # if the number of requested cells is larger than the size of the
                                               ## presynaptic population, we allow multiple connections for a given cell
                create = numpy.concatenate((create, self.projection.rng.permutation(precreate)))
            create = create[:n_connections]
        else:
            create = precreate            
        targets = self.candidates[create]        
        weights = self.weights_generator.get(self.N, self.distance_matrix, create)
        delays  = self.delays_generator.get(self.N, self.distance_matrix, create)        
        
        if len(targets) > 0:
            self.projection._divergent_connect(src, targets.tolist(), weights, delays)
        
    
class AllToAllConnector(Connector):
    """
    Connects all cells in the presynaptic population to all cells in the
    postsynaptic population.
    """
    parameter_names = ('allow_self_connections',)
    
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
        self.progressbar(len(projection.pre))
        for count, src in enumerate(projection.pre.all()):
            connector._probabilistic_connect(src, 1)
            self.progression(count, projection._simulator.state.mpi_rank)
            
    

class FixedProbabilityConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability is constant.
    """
    parameter_names = ('allow_self_connections', 'p_connect')
    
    def __init__(self, p_connect, allow_self_connections=True, weights=0.0,
                 delays=None, space=Space(), safe=True, verbose=False):
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
        #assert projection.rng.parallel_safe
        connector = ProbabilisticConnector(projection, self.weights, self.delays,
                                           self.allow_self_connections, self.space,
                                           safe=self.safe)
        self.progressbar(len(projection.pre))
        for count, src in enumerate(projection.pre.all()):
            connector._probabilistic_connect(src, self.p_connect)
            self.progression(count, projection._simulator.state.mpi_rank)
            

class DistanceDependentProbabilityConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability depends on distance.
    """
    parameter_names = ('allow_self_connections', 'd_expression')
    
    def __init__(self, d_expression, allow_self_connections=True,
                 weights=0.0, delays=None, space=Space(), safe=True, verbose=False, n_connections=None):
        """
        Create a new connector.
        
        `d_expression` -- the right-hand side of a valid python expression for
            probability, involving 'd', e.g. "exp(-abs(d))", or "d<3"
        `n_connections`  -- The number of efferent synaptic connections per neuron.                 
        `space` -- a Space object.
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created, or a distance expression as for `d_expression`. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        Connector.__init__(self, weights, delays, space, safe, verbose)
        assert isinstance(d_expression, str) or callable(d_expression)
        try:
            if isinstance(d_expression, str) and not expand_distances(d_expression):                       
                d = 0; assert 0 <= eval(d_expression), eval(d_expression)
                d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
        except ZeroDivisionError, err:
            raise ZeroDivisionError("Error in the distance expression %s. %s" % (d_expression, err))
        self.d_expression = d_expression
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        self.n_connections          = n_connections        
        
    def connect(self, projection):
        """Connect-up a Projection."""
        connector       = ProbabilisticConnector(projection, self.weights, self.delays, self.allow_self_connections, self.space, safe=self.safe)
        proba_generator = ProbaGenerator(self.d_expression, connector.local)
        self.progressbar(len(projection.pre))
        if (projection._simulator.state.num_processes > 1) and (self.n_connections is not None):
            raise Exception("n_connections not implemented yet for this connector in parallel !")

        for count, src in enumerate(projection.pre.all()):     
            connector.distance_matrix.set_source(src.position)
            proba  = proba_generator.get(connector.N, connector.distance_matrix)
            if proba.dtype == 'bool':
                proba = proba.astype(float)
            connector._probabilistic_connect(src, proba, self.n_connections)
            self.progression(count, projection._simulator.state.mpi_rank)
    

class FromListConnector(Connector):
    """
    Make connections according to a list.
    """
    parameter_names = ('conn_list',)
    
    def __init__(self, conn_list, safe=True, verbose=False):
        """
        Create a new connector.
        
        `conn_list` -- a list of tuples, one tuple for each connection. Each
                       tuple should contain:
                          (pre_idx, post_idx, weight, delay)
                       where pre_idx is the index (i.e. order in the Population,
                       not the ID) of the presynaptic neuron, and post_idx is
                       the index of the postsynaptic neuron.
        """
        # needs extending for dynamic synapses.
        Connector.__init__(self, 0.0, None, safe=safe, verbose=verbose)
        self.conn_list  = numpy.array(conn_list)               
        
    def connect(self, projection):
        """Connect-up a Projection."""
        idx     = numpy.argsort(self.conn_list[:, 0])
        self.sources    = numpy.unique(self.conn_list[:,0]).astype(numpy.int)
        self.candidates = projection.post.local_cells
        self.conn_list  = self.conn_list[idx]
        self.progressbar(len(self.sources))        
        count = 0
        left  = numpy.searchsorted(self.conn_list[:,0], self.sources, 'left')
        right = numpy.searchsorted(self.conn_list[:,0], self.sources, 'right')
        #tests = "|".join(['(tgts == %d)' %id for id in self.candidates])
        for src, l, r in zip(self.sources, left, right):
            targets = self.conn_list[l:r, 1].astype(numpy.int)
            weights = self.conn_list[l:r, 2]
            delays  = self.conn_list[l:r, 3]
            try:
                src     = projection.pre.all_cells[src]
            except IndexError:
                raise errors.ConnectionError("invalid source index %s" % src)
            try:
                tgts    = projection.post.all_cells[targets]
            except IndexError:
                raise errors.ConnectionError("invalid target index or indices")
            ## We need to exclude the non local cells. Fastidious, need maybe
            ## to use a convergent_connect method, instead of a divergent_connect one
            #idx     = eval(tests)
            #projection._divergent_connect(src, tgts[idx].tolist(), weights[idx], delays[idx])
            projection._divergent_connect(src, tgts.tolist(), weights, delays)
            self.progression(count, projection._simulator.state.mpi_rank)
            count += 1
        
class FromFileConnector(FromListConnector):
    """
    Make connections according to a list read from a file.
    """
    parameter_names = ('filename', 'distributed')
    
    def __init__(self, file, distributed=False, safe=True, verbose=False):
        """
        Create a new connector.
        
        `file`        -- file object containing a list of connections, in
                         the format required by `FromListConnector`.
        `distributed` -- if this is True, then each node will read connections
                         from a file called `filename.x`, where `x` is the MPI
                         rank. This speeds up loading connections for
                         distributed simulations.
        """
        Connector.__init__(self, 0.0, None, safe=safe, verbose=verbose)
        
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='r')        
        self.file        = file
        self.distributed = distributed

    def connect(self, projection):
        """Connect-up a Projection."""
        if self.distributed:
            self.file.rename("%s.%d" % (self.file.name, projection._simulator.state.mpi_rank))        
        self.conn_list = self.file.read()
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
    parameter_names = ('allow_self_connections', 'n')
    
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
        if self.delays is None:
            self.delays = projection._simulator.state.min_delay
        local             = numpy.ones(len(projection.post), bool)
        weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
        delays_generator  = DelayGenerator(self.delays, local, kernel=projection._simulator.state, safe=self.safe)
        distance_matrix   = DistanceMatrix(projection.post.positions, self.space)
        candidates        = projection.post.all_cells
        size              = len(projection.post)    
        self.progressbar(len(projection.pre))
        
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Use of NativeRNG not implemented.")        
        
        for count, src in enumerate(projection.pre.all()):
            # pick n neurons at random
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            
            idx = numpy.arange(size)
            if not self.allow_self_connections and projection.pre == projection.post:                
                i   = numpy.where(candidates == src)[0]
                idx = numpy.delete(idx, i)

            create = numpy.array([], dtype=numpy.int)          
            while len(create) < n: # if the number of requested cells is larger than the size of the
                                    # postsynaptic population, we allow multiple connections for a given cell   
                create = numpy.concatenate((create, projection.rng.permutation(idx)[:n]))                 

            distance_matrix.set_source(src.position)
            create  = create[:n].astype(numpy.int)
            targets = candidates[create]
            weights = weights_generator.get(n, distance_matrix, create)
            delays  = delays_generator.get(n, distance_matrix, create)
               
            if len(targets) > 0:
                projection._divergent_connect(src, targets.tolist(), weights, delays)
            
            self.progression(count, projection._simulator.state.mpi_rank)
        

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
    parameter_names = ('allow_self_connections', 'n')
    
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
        if self.delays is None:
            self.delays = projection._simulator.state.min_delay
        local             = numpy.ones(len(projection.pre), bool)
        weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
        delays_generator  = DelayGenerator(self.delays, local, kernel=projection._simulator.state, safe=self.safe)
        distance_matrix   = DistanceMatrix(projection.pre.positions, self.space)              
        candidates        = projection.pre.all_cells 
        size              = len(projection.pre)
        self.progressbar(len(projection.post.local_cells))
        
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Warning: use of NativeRNG not implemented.")
            
        for count, tgt in enumerate(projection.post.local_cells):
            # pick n neurons at random
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            
            idx        = numpy.arange(size, dtype=numpy.int)
            if not self.allow_self_connections and projection.pre == projection.post:
                i   = numpy.where(candidates == tgt)[0]
                idx = numpy.delete(idx, i)
            create = numpy.array([], dtype=numpy.int) 
            while len(create) < n: # if the number of requested cells is larger than the size of the
                                    # presynaptic population, we allow multiple connections for a given cell
                create = numpy.concatenate((create, projection.rng.permutation(idx)[:n]))
                                             
            distance_matrix.set_source(tgt.position)
            create  = create[:n].astype(numpy.int)
            sources = candidates[create]
            weights = weights_generator.get(n, distance_matrix, create)
            delays  = delays_generator.get(n, distance_matrix, create)            
                                            
            for src, w, d in zip(sources, weights, delays):
                projection._divergent_connect(src, tgt, w, d)
            
            self.progression(count, projection._simulator.state.mpi_rank)
        

class OneToOneConnector(Connector):
    """
    Where the pre- and postsynaptic populations have the same size, connect
    cell i in the presynaptic population to cell i in the postsynaptic
    population for all i.
    """
    parameter_names = tuple()
    
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
        if self.delays is None:
            self.delays = projection._simulator.state.min_delay
        if projection.pre.size == projection.post.size:
            N                 = projection.post.size
            local             = projection.post._mask_local
            if isinstance(self.weights, basestring) or isinstance(self.delays, basestring):
                raise Exception('Expression for weights or delays is not supported for OneToOneConnector !')
            weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
            delays_generator  = DelayGenerator(self.delays, local, kernel=projection._simulator.state, safe=self.safe)                
            weights           = weights_generator.get(N)
            delays            = delays_generator.get(N)
            self.progressbar(len(projection.post.local_cells))                        
            count             = 0            
            create            = numpy.arange(N, dtype=numpy.int)[local]
            sources           = projection.pre.all_cells[create] 
                        
            for tgt, src, w, d in zip(projection.post.local_cells, sources, weights, delays):
                # the float is in case the values are of type numpy.float64, which NEST chokes on
                projection._divergent_connect(src, [tgt], [float(w)], [float(d)])
                self.progression(count, projection._simulator.state.mpi_rank)
                count += 1
        else:
            raise errors.InvalidDimensionsError("OneToOneConnector does not support presynaptic and postsynaptic Populations of different sizes.")


class SmallWorldConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability depends on distance.
    """
    parameter_names = ('allow_self_connections', 'degree', 'rewiring')
    
    def __init__(self, degree, rewiring, allow_self_connections=True,
                 weights=0.0, delays=None, space=Space(), safe=True, verbose=False, n_connections=None):
        """
        Create a new connector.
        
        `degree` -- the region lenght where nodes will be connected locally
        `rewiring` -- the probability of rewiring each eadges 
        `space` -- a Space object.
        `allow_self_connections` -- if the connector is used to connect a
            Population to itself, this flag determines whether a neuron is
            allowed to connect to itself, or only to other neurons in the
            Population.        
        `n_connections`  -- The number of efferent synaptic connections per neuron. 
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
        self.n_connections          = n_connections
        
    def _smallworld_connect(self, src, p, n_connections=None):
        """
        Connect-up a Projection with connection probability p, where p may be either
        a float 0<=p<=1, or a dict containing a float array for each pre-synaptic
        cell, the array containing the connection probabilities for all the local
        targets of that pre-synaptic cell.
        """
        rarr = self.probas_generator.get(self.N)
        if not core.is_listlike(rarr) and numpy.isscalar(rarr): # if N=1, rarr will be a single number
            rarr = numpy.array([rarr])
        precreate = numpy.where(rarr < p)[0]  
        self.distance_matrix.set_source(src.position)        
        
        if not self.allow_self_connections and self.projection.pre == self.projection.post:
            i         = numpy.where(self.candidates == src)[0]
            precreate = numpy.delete(precreate, i)        
        
        idx = numpy.arange(self.size, dtype=numpy.int)
        if not self.allow_self_connections and self.projection.pre == self.projection.post:
            i   = numpy.where(self.candidates == src)[0]
            idx = numpy.delete(idx, i)
        
        rarr    = self.probas_generator.get(self.N)[precreate]
        rewired = numpy.where(rarr < self.rewiring)[0]
        N       = len(rewired)
        if N > 0:
            new_idx            = (len(idx)-1) * self.probas_generator.get(self.N)[precreate]
            precreate[rewired] = idx[new_idx.astype(int)]
        
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
                    
        if len(targets) > 0:
            self.projection._divergent_connect(src, targets.tolist(), weights, delays)

    def connect(self, projection):
        """Connect-up a Projection."""
        if self.delays is None:
            self.delays = projection._simulator.state.min_delay
        local                  = numpy.ones(len(projection.post), bool)
        self.N                 = projection.post.size        
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Use of NativeRNG not implemented.")
        else:
            self.rng = projection.rng
        self.weights_generator = WeightGenerator(self.weights, local, projection, self.safe)
        self.delays_generator  = DelayGenerator(self.delays, local, kernel=projection._simulator.state, safe=self.safe)
        self.probas_generator  = ProbaGenerator(RandomDistribution('uniform',(0,1), rng=self.rng), local)
        self.distance_matrix   = DistanceMatrix(projection.post.positions, self.space, local)
        self.projection        = projection
        self.candidates        = projection.post.all_cells          
        self.size              = len(projection.post)
        self.progressbar(len(projection.pre))        
        proba_generator = ProbaGenerator(self.d_expression, local)
        for count, src in enumerate(projection.pre.all()):     
            self.distance_matrix.set_source(src.position)
            proba = proba_generator.get(self.N, self.distance_matrix).astype(float)
            self._smallworld_connect(src, proba, self.n_connections)
            self.progression(count, projection._simulator.state.mpi_rank)


class CSAConnector(Connector):
    parameter_names = ('cset',)
    
    if haveCSA:
        def __init__ (self, cset, weights=None, delays=None, safe=True, verbose=False):
            """
            """
            Connector.__init__(self, None, None, safe=safe, verbose=verbose)
            self.cset = cset
            if csa.arity(cset) == 0:
                #assert weights is not None and delays is not None, \
                #       'must specify weights and delays in addition to a CSA mask'
                self.weights = weights
                if weights is None:
                    self.weights = DEFAULT_WEIGHT
                self.delays = delays
            else:
                assert csa.arity(cset) == 2, 'must specify mask or connection-set with arity 2'
                assert weights is None and delays is None, \
                       "weights or delays specified both in connection-set and as CSAConnector argument"
    else:
        def __init__ (self, cset, safe=True, verbose=False):
            raise RuntimeError, "CSAConnector not available---couldn't import csa module"

    @staticmethod
    def isConstant (x):
        return isinstance (x, (int, float))
    
    @staticmethod
    def constantIterator (x):
        while True:
            yield x

    def connect(self, projection):
        """Connect-up a Projection."""
        if self.delays is None:
            self.delays = projection._simulator.state.min_delay
        # Cut out finite part
        c = csa.cross((0, projection.pre.size-1), (0, projection.post.size-1)) * self.cset
        
        if csa.arity(self.cset) == 2:
            # Connection-set with arity 2
            for (i, j, weight, delay) in c:
                projection._divergent_connect(projection.pre[i], [projection.post[j]], weight, delay)
        elif CSAConnector.isConstant (self.weights) \
             and CSAConnector.isConstant (self.delays):
            # Mask with constant weights and delays
            for (i, j) in c:
                projection._divergent_connect (projection.pre[i], [projection.post[j]], self.weights, self.delays)
        else:
            # Mask with weights and/or delays iterable
            weights = self.weights
            if CSAConnector.isConstant (weights):
                weights = CSAConnector.constantIterator (weights)
            delays = self.delays
            if CSAConnector.isConstant (delays):
                delays = CSAConnector.constantIterator (delays)
            for (i, j), weight, delay in zip (c, weights, delays):
                projection._divergent_connect (projection.pre[i], [projection.post[j]], weight, delay)
