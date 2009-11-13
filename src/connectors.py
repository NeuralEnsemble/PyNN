# encoding: utf-8
"""
Base Connector classes with default implementation.

    AllToAllConnector
    OneToOneConnector
    FixedProbabilityConnector
    DistanceDependentProbabilityConnector
    FixedNumberPreConnector
    FixedNumberPostConnector
    FromListConnector
    FromFileConnector

$Id$
"""

import logging
import numpy
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh
from pyNN import random, common

logger = logging.getLogger("PyNN")

def _probabilistic_connect(connector, projection, p):
    """
    Connect-up a Projection with connection probability p, where p may be either
    a float 0<=p<=1, or a dict containing a float array for each pre-synaptic
    cell, the array containing the connection probabilities for all the local
    targets of that pre-synaptic cell.
    """
    if isinstance(projection.rng, random.NativeRNG):
        raise Exception("Use of NativeRNG not implemented.")
    else:
        rng = projection.rng
    
    local = projection.post._mask_local.flatten()
    is_conductance = common.is_conductance(projection.post.index(0))
    for src in projection.pre.all():
        # ( the following two lines are a nice idea, but this needs some thought for
        #   the parallel case, to ensure reproducibility when varying the number
        #   of processors
        #      N = rng.binomial(npost,self.p_connect,1)[0]
        #      targets = sample(postsynaptic_neurons, N)   # )
        N = projection.post.size
        # if running in parallel, rng.next(N) will not return N values, but only
        # as many as are needed on this node, as determined by mask_local.
        # Over the simulation as a whole (all nodes), N values will indeed be
        # returned.
        rarr = rng.next(N, 'uniform', (0, 1), mask_local=local)
        if not common.is_listlike(rarr) and common.is_number(rarr): # if N=1, rarr will be a single number
            rarr = numpy.array([rarr])
        if common.is_number(p):
            create = rarr < p
        else:
            create = rarr < p[src][local]
        if create.shape != projection.post.local_cells.shape:
            logger.warning("Too many random numbers. Discarding the excess. Did you specify MPI rank and number of processes when you created the random number generator?")
            create = create[:projection.post.local_cells.size]
        targets = projection.post.local_cells[create].tolist()
        
        weights = connector.get_weights(N, local)[create]
        weights = common.check_weight(weights, projection.synapse_type, is_conductance)
        delays  = connector.get_delays(N, local)[create]
        
        if not connector.allow_self_connections and projection.pre == projection.post and src in targets:
            assert len(targets) == len(weights) == len(delays)
            i = targets.index(src)
            weights = numpy.delete(weights, i)
            delays = numpy.delete(delays, i)
            targets.remove(src)
        
        if len(targets) > 0:
            projection.connection_manager.connect(src, targets, weights, delays)


class AllToAllConnector(common.Connector):
    """
    Connects all cells in the presynaptic population to all cells in the
    postsynaptic population.
    """
    
    def __init__(self, allow_self_connections=True, weights=0.0, delays=None):
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
        """
        common.Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections

    def connect(self, projection):
        """Connect-up a Projection."""
        _probabilistic_connect(self, projection, 1.0)


class FromListConnector(common.Connector):
    """
    Make connections according to a list.
    """
    
    def __init__(self, conn_list):
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
        common.Connector.__init__(self, 0., 0.)
        self.conn_list = conn_list        
        
    def connect(self, projection):
        """Connect-up a Projection."""
        # slow: should maybe sort by pre
        for i in xrange(len(self.conn_list)):
            src, tgt, weight, delay = self.conn_list[i][:]
            src = projection.pre[tuple(src)]           
            tgt = projection.post[tuple(tgt)]
            projection.connection_manager.connect(src, [tgt], weight, delay)
    

class FromFileConnector(FromListConnector):
    """
    Make connections according to a list read from a file.
    """
    
    def __init__(self, filename, distributed=False):
        """
        Create a new connector.
        
        `filename` -- name of a text file containing a list of connections, in
                      the format required by `FromListConnector`.
        `distributed` -- if this is True, then each node will read connections
                         from a file called `filename.x`, where `x` is the MPI
                         rank. This speeds up loading connections for
                         distributed simulations.
        """
        common.Connector.__init__(self, 0., 0.)
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
        
        
class FixedNumberPostConnector(common.Connector):
    """
    Each pre-synaptic neuron is connected to exactly n post-synaptic neurons
    chosen at random.
    
    If n is less than the size of the post-synaptic population, there are no
    multiple connections, i.e., no instances of the same pair of neurons being
    multiply connected. If n is greater than the size of the post-synaptic
    population, all possible single connections are made before starting to add
    duplicate connections.
    """
    
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=None):
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
        common.Connector.__init__(self, weights, delays)
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
        if isinstance(projection.rng, random.NativeRNG):
            raise Exception("Warning: use of NativeRNG not implemented.")
            
        for source in projection.pre.all_cells.flat:
            # pick n neurons at random
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n

            candidates = projection.post.all_cells.flatten().tolist()
            if not self.allow_self_connections and projection.pre == projection.post:
                candidates.remove(source)
            targets = []
            while len(targets) < n: # if the number of requested cells is larger than the size of the
                                    # postsynaptic population, we allow multiple connections for a given cell
                targets += [candidates[candidates.index(id)] for id in projection.rng.permutation(candidates)[0:n]]
                # have to use index() because rng.permutation returns ints, not ID objects
            
            targets = numpy.array(targets[:n], dtype=common.IDMixin)
            
            weights = self.get_weights(n)
            is_conductance = common.is_conductance(projection.post.index(0))
            weights = common.check_weight(weights, projection.synapse_type, is_conductance)
            delays = self.get_delays(n)
            
            #local = numpy.array([tgt.local for tgt in targets])
            #if local.size > 0:
            #    targets = targets[local]
            #    weights = weights[local]
            #    delays = delays[local]
            targets = targets.tolist()
            #print common.rank(), source, targets
            if len(targets) > 0:
                projection.connection_manager.connect(source, targets, weights, delays)
                    

class FixedNumberPreConnector(common.Connector):
    """
    Each post-synaptic neuron is connected to exactly n pre-synaptic neurons
    chosen at random.
    
    If n is less than the size of the pre-synaptic population, there are no
    multiple connections, i.e., no instances of the same pair of neurons being
    multiply connected. If n is greater than the size of the pre-synaptic
    population, all possible single connections are made before starting to add
    duplicate connections.
    """
    
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=None):
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
        common.Connector.__init__(self, weights, delays)
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
            
            for source, w, d in zip(sources, weights, delays):
                projection.connection_manager.connect(source, [target], w, d)
                    

class OneToOneConnector(common.Connector):
    """
    Where the pre- and postsynaptic populations have the same size, connect
    cell i in the presynaptic population to cell i in the postsynaptic
    population for all i.
    """
    #In fact, despite the name, this should probably be generalised to the
    #case where the pre and post populations have different dimensions, e.g.,
    #cell i in a 1D pre population of size n should connect to all cells
    #in row i of a 2D post population of size (n,m).
    
    
    def __init__(self, weights=0.0, delays=None):
        """
        Create a new connector.
        
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        common.Connector.__init__(self, weights, delays)
    
    def connect(self, projection):
        """Connect-up a Projection."""
        if projection.pre.dim == projection.post.dim:
            N = projection.post.size
            local = projection.post._mask_local.flatten()
            weights = self.get_weights(N, local)
            is_conductance = common.is_conductance(projection.post.index(0))
            weights = common.check_weight(weights, projection.synapse_type, is_conductance)
            delays = self.get_delays(N, local)
            
            for tgt, w, d in zip(projection.post.local_cells,
                                 weights,
                                 delays):
                src = projection.pre.index(projection.post.id_to_index(tgt))
                
                # the float is in case the values are of type numpy.float64, which NEST chokes on
                projection.connection_manager.connect(src, [tgt], float(w), float(d))
        else:
            raise common.InvalidDimensionsError("OneToOneConnector does not support presynaptic and postsynaptic Populations of different sizes.")


class FixedProbabilityConnector(common.Connector):
    """
    For each pair of pre-post cells, the connection probability is constant.
    """
    
    def __init__(self, p_connect, allow_self_connections=True, weights=0.0, delays=None):
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
        """
        common.Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        self.p_connect = float(p_connect)
        assert 0 <= self.p_connect
    
    def connect(self, projection):
        """Connect-up a Projection."""
        logger.info("Connecting %s to %s with probability %s" % (projection.pre.label,
                                                                  projection.post.label,
                                                                  self.p_connect))
        _probabilistic_connect(self, projection, self.p_connect)    

        
class DistanceDependentProbabilityConnector(common.Connector):
    """
    For each pair of pre-post cells, the connection probability depends on distance.
    """
    
    AXES = {'x' : [0],    'y': [1],    'z': [2],
            'xy': [0,1], 'yz': [1,2], 'xz': [0,2], 'xyz': None, None: None}
    
    def __init__(self, d_expression, axes=None, scale_factor=1.0, offset=0.0,
                 periodic_boundaries=None, allow_self_connections=True,
                 weights=0.0, delays=None):
        """
        Create a new connector.
        
        d_expression -- the right-hand side of a valid python expression for
            probability, involving 'd', e.g. "exp(-abs(d))", or "d<3"
        axes -- if not supplied, then the 3D distance is calculated. If supplied,
            axes should be a string containing the axes to be used, e.g. 'x', or
            'yz'. axes='xyz' is the same as axes=None.
        scale_factor -- it may be that the pre and post populations use
            different units for position, e.g. degrees and µm. In this case,
            `scale_factor` can be specified, which is applied to the positions
            in the post-synaptic population.
        offset -- if the origins of the coordinate systems of the pre- and post-
            synaptic populations are different, `offset` can be used to adjust
            for this difference.
        periodic_boundaries -- either `None`, or a tuple giving the distance at
            which the boundaries wrap round for each dimension.
        `weights` -- may either be a float, a RandomDistribution object, a list/
                     1D array with at least as many items as connections to be
                     created. Units nA.
        `delays`  -- as `weights`. If `None`, all synaptic delays will be set
                     to the global minimum delay.
        """
        common.Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        assert isinstance(d_expression, str)
        try:
            d = 0; assert 0 <= eval(d_expression), eval(d_expression)
            d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
        except ZeroDivisionError, err:
            raise ZeroDivisionError("Error in the distance expression %s. %s" % (d_expression, err))
        self.d_expression = d_expression
        self.allow_self_connections = allow_self_connections
        self.mask = DistanceDependentProbabilityConnector.AXES[axes]
        self.periodic_boundaries = periodic_boundaries
        if self.mask is not None:
            self.mask = numpy.array(self.mask)
        self.scale_factor = scale_factor
        self.offset = offset
        if isinstance(self.weights, basestring):
            self.w_expr = self.weights
            self.weights = numpy.empty((1,0))
        if isinstance(self.delays, basestring):
            self.d_expr = self.delays
            self.delays = numpy.empty((1,0))
        
    def connect(self, projection):
        """Connect-up a Projection."""
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is True:
            dimensions = projection.post.dim
            periodic_boundaries = tuple(numpy.concatenate((dimensions, numpy.zeros(3-len(dimensions)))))
        if periodic_boundaries:
            logger.info("Periodic boundaries activated and set to size %s" % str(periodic_boundaries))
        p = {}
        
        for src in projection.pre.all():
            d = common.distances(src, projection.post, self.mask,
                                 self.scale_factor, self.offset,
                                 periodic_boundaries)
            p[src] = eval(self.d_expression).flatten()
            if p[src].dtype == 'bool':
                p[src] = p[src].astype(float)
            if hasattr(self, 'w_expr'):
                self.weights = numpy.concatenate((self.weights, eval(self.w_expr)), axis=1)
            if hasattr(self, 'd_expr'):
                self.delays = numpy.concatenate((self.delays, eval(self.d_expr)), axis=1)
        
        if hasattr(self.weights, 'shape'): self.weights = self.weights.flatten()
        if hasattr(self.delays, 'shape'): self.delays = self.delays.flatten()
        _probabilistic_connect(self, projection, p)
