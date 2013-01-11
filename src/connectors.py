"""
Defines a common implementation of the built-in PyNN Connector classes.

Simulator modules may use these directly, or may implement their own versions
for improved performance.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from __future__ import division
from pyNN.space import Space
from pyNN.core import LazyArray
from pyNN.random import RandomDistribution
from pyNN.common.populations import is_conductance
from pyNN import errors, descriptions
from pyNN.recording import files
import numpy
from itertools import izip
import logging

from lazyarray import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, exp, \
                      fabs, floor, fmod, hypot, ldexp, log, log10, modf, power, \
                      sin, sinh, sqrt, tan, tanh, maximum, minimum
from numpy import e, pi

try:
    import csa
    haveCSA = True
except ImportError:
    haveCSA = False

logger = logging.getLogger("PyNN")

DEFAULT_WEIGHT = 0.0


def check_weights(weights, synapse_sign, is_conductance):
    # if projection.post is an Assembly, some components might have cond-synapses, others curr, so need a more sophisticated check here
    # consider moving checks inside the lazyarray module?
    if weights is None:
        weights = DEFAULT_WEIGHT
    if isinstance(weights, numpy.ndarray):
        all_negative = (weights <= 0).all()
        all_positive = (weights >= 0).all()
        if not (all_negative or all_positive):
            raise errors.ConnectionError("Weights must be either all positive or all negative")
    elif numpy.isreal(weights):
        all_positive = weights >= 0
        all_negative = weights < 0
    else:
        raise errors.ConnectionError("Weights must be a number or an array of numbers.")
    if is_conductance or synapse_sign == 'excitatory':
        if not all_positive:
            raise errors.ConnectionError("Weights must be positive for conductance-based and/or excitatory synapses")
    elif is_conductance == False and synapse_sign == 'inhibitory':
        if not all_negative:
            raise errors.ConnectionError("Weights must be negative for current-based, inhibitory synapses")
    else:  # This should never happen.
        raise Exception("Can't check weight, conductance status unknown.")
    return weights


def check_delays(delays, min_delay, max_delay):
    if isinstance(delays, numpy.ndarray):
        below_max = (delays <= max_delay).all()
        above_min = (delays >= min_delay).all()
        in_range = below_max and above_min
    elif numpy.isreal(delays):
        in_range = min_delay <= delays <= max_delay
    else:
        raise errors.ConnectionError("Delays must be a number or an array of numbers.")
    if not in_range:
        raise errors.ConnectionError("Delay (%s) is out of range [%s, %s]" % (delays, min_delay, max_delay))
    return delays


class Connector(object):
    """
    Base class for connectors.

    All connector sub-classes have the following optional keyword arguments:
        `space`:
            a `Space` object, needed if you wish to specify distance-dependent
            weights or delays.
        `safe`:
            if True, check that weights and delays have valid values. If False,
            this check is skipped.
        `callback`:
            a function that will be called with the fractional progress of the
            connection routine. An example would be `progress_bar.set_level`.
    """

    def __init__(self, space=Space(), safe=True, callback=None):
        """
        docstring needed
        """
        self.space   = space
        self.safe    = safe
        self.callback = callback
        if callback is not None:
            assert callable(callback)

    def connect(self, projection):
        raise NotImplementedError()

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
                   'parameters': self.get_parameters()}
        return descriptions.render(engine, template, context)


class MapConnector(Connector):
    """

    """
    # abstract base class

    def _generate_distance_map(self, projection):
        position_generators = (projection.pre.position_generator,
                               projection.post.position_generator)
        return LazyArray(self.space.distance_generator3D(*position_generators),
                         shape=projection.shape)

    def _connect_with_map(self, projection, connection_map, distance_map=None):
        logger.debug("Connecting %s using a connection map" % projection.label)
        if distance_map is None:
            distance_map = self._generate_distance_map(projection)
#        # If any of the maps are based on parallel-safe random number generators,
#        # we need to iterate over all post-synaptic cells, so we can generate then
#        # throw away the random numbers for the non-local nodes.
#        # Otherwise, we only need to iterate over local post-synaptic cells.
#        def parallel_safe(map):
#            return (isinstance(map.base_value, RandomDistribution) and
#                    map.base_value.rng.parallel_safe)
        column_indices = numpy.arange(projection.post.size)
#        if parallel_safe(weight_map) or parallel_safe(delay_map): # TODO check all the plasticity maps as well
#            logger.debug("Parallel-safe iteration.")
#            components = (
#                column_indices,
#                projection.post.all_cells,
#                connection_map.by_column())
#        else:
        if True: ###
            mask = projection.post._mask_local
            components = (
                column_indices[mask],
                connection_map.by_column(mask))
        for count, (col, source_mask) in enumerate(izip(*components)):
            if source_mask is True or source_mask.any():
                if source_mask is True:
                    source_mask = numpy.arange(projection.pre.size, dtype=int)
                else:
                    source_mask = source_mask.nonzero()[0]  # bool to integer mask
                connection_parameters = {}
                for name, map in projection.synapse_type.translated_parameters.items():
                    map.shape = (projection.pre.size, projection.post.size)
                    if callable(map.base_value):  # map is assumed to be a function of "d"
                        map = map(distance_map)
                    if map.is_homogeneous:
                        connection_parameters[name] = map.evaluate(simplify=True)
                    else:
                        connection_parameters[name] = map[source_mask, col]
                        
                #logger.debug("Convergent connect %d neurons to #%s, delays in range (%g, %g)" % (sources.size, tgt, delays.min(), delays.max()))
#                if self.safe:
#                    # (might be cheaper to do the weight and delay check before evaluating the larray)
#                    weights = check_weights(weights, projection.synapse_type, is_conductance(projection.post.local_cells[0]))
#                    delays = check_delays(delays,
#                                          projection._simulator.state.min_delay,
#                                          projection._simulator.state.max_delay)
#                    # TODO: add checks for plasticity parameters
#                #logger.debug("mask: %s, w: %s, d: %s", source_mask, weights, delays)
                if True:
                    projection._convergent_connect(source_mask, col, **connection_parameters)
                    if self.callback:
                        self.callback(count/projection.post.local_size)


class AllToAllConnector(MapConnector):
    """
    Connects all cells in the presynaptic population to all cells in the
    postsynaptic population.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
    """
    parameter_names = ('allow_self_connections',)

    def __init__(self, allow_self_connections=True, space=Space(), safe=True,
                 callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, space, safe, callback)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections

    def connect(self, projection):
        if not self.allow_self_connections and projection.pre == projection.post:
            connection_map = LazyArray(lambda i,j: i != j, shape=projection.shape)
        else:
            connection_map = LazyArray(True, shape=projection.shape)
        self._connect_with_map(projection, connection_map)


class FixedProbabilityConnector(MapConnector):
    """
    For each pair of pre-post cells, the connection probability is constant.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `p_connect`:
            a float between zero and one. Each potential connection is created
            with this probability.
        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
    """
    parameter_names = ('allow_self_connections', 'p_connect')

    def __init__(self, p_connect, allow_self_connections=True, space=Space(),
                 safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, space, safe, callback)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        self.p_connect = float(p_connect)
        assert 0 <= self.p_connect

    def connect(self, projection):
        random_map = LazyArray(RandomDistribution('uniform', (0, 1), rng=projection.rng),
                               projection.shape)
        connection_map = random_map < self.p_connect
        if not self.allow_self_connections and projection.pre == projection.post:
            connection_map *= LazyArray(lambda i,j: i != j, shape=projection.shape)
        self._connect_with_map(projection, connection_map)


class DistanceDependentProbabilityConnector(MapConnector):
    """
    For each pair of pre-post cells, the connection probability depends on distance.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `d_expression`:
            the right-hand side of a valid Python expression for probability,
            involving 'd', e.g. "exp(-abs(d))", or "d<3"
        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
    """
    parameter_names = ('allow_self_connections', 'd_expression')

    def __init__(self, d_expression, allow_self_connections=True,
                 space=Space(), safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, space, safe, callback)
        assert isinstance(d_expression, str) or callable(d_expression)
        assert isinstance(allow_self_connections, bool)
        try:
            if isinstance(d_expression, str):
                d = 0; assert 0 <= eval(d_expression), eval(d_expression)
                d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
        except ZeroDivisionError, err:
            raise ZeroDivisionError("Error in the distance expression %s. %s" % (d_expression, err))
        self.d_expression = d_expression
        self.allow_self_connections = allow_self_connections
        self.distance_function = eval("lambda d: %s" % self.d_expression)

    def connect(self, projection):
        distance_map = self._generate_distance_map(projection)
        probability_map = self.distance_function(distance_map)
        random_map = LazyArray(RandomDistribution('uniform', (0, 1), rng=projection.rng),
                               projection.shape)
        connection_map = random_map < probability_map
        if not self.allow_self_connections and projection.pre == projection.post:
            connection_map *= LazyArray(lambda i,j: i != j, shape=projection.shape)
        self._connect_with_map(projection, connection_map, distance_map)


class FromListConnector(Connector):
    """
    Make connections according to a list.

    Arguments:
        `conn_list`:
            a list of tuples, one tuple for each connection. Each tuple should contain:
            `(pre_idx, post_idx, weight, delay)` where `pre_idx` is the index
            (i.e. order in the Population, not the ID) of the presynaptic
            neuron, and `post_idx` is the index of the postsynaptic neuron.
        `safe`:
            if True, check that weights and delays have valid values. If False,
            this check is skipped.
        `callback`:
            if True, display a progress bar on the terminal.
    """
    parameter_names = ('conn_list',)

    def __init__(self, conn_list, safe=True, callback=None):
        """
        Create a new connector.
        """
        # needs extending for dynamic synapses.
        Connector.__init__(self, safe=safe, callback=callback)
        self.conn_list  = numpy.array(conn_list)

    def connect(self, projection):
        """Connect-up a Projection."""
        logger.debug("self.conn_list = \n%s", self.conn_list)
        # need to do some profiling, to figure out the best way to do this:
        #  - order of sorting/filtering by local
        #  - use numpy.unique, or just do in1d(self.conn_list)?
        idx  = numpy.argsort(self.conn_list[:, 1])
        targets = numpy.unique(self.conn_list[:, 1]).astype(numpy.int)
        local = numpy.in1d(targets,
                           numpy.arange(projection.post.size)[projection.post._mask_local],
                           assume_unique=True)
        local_targets = targets[local]
        self.conn_list = self.conn_list[idx]
        left  = numpy.searchsorted(self.conn_list[:, 1], local_targets, 'left')
        right = numpy.searchsorted(self.conn_list[:, 1], local_targets, 'right')
        logger.debug("idx = %s", idx)
        logger.debug("targets = %s", targets)
        logger.debug("local_targets = %s", local_targets)
        logger.debug("self.conn_list = \n%s", self.conn_list)
        logger.debug("left = %s", left)
        logger.debug("right = %s", right)

        for tgt, l, r in zip(local_targets, left, right):
            sources = self.conn_list[l:r, 0].astype(numpy.int)
            weights = self.conn_list[l:r, 2]
            delays  = self.conn_list[l:r, 3]
            try:
                srcs = projection.pre.all_cells[sources]
            except IndexError:
                raise errors.ConnectionError("invalid sources index or indices")
            try:
                tgt = projection.post.all_cells[tgt]
            except IndexError:
                raise errors.ConnectionError("invalid target index %d" % tgt)
            projection._convergent_connect(srcs, tgt, weight=weights, delay=delays)


class FromFileConnector(FromListConnector):
    """
    Make connections according to a list read from a file.

    Arguments:
        `file`:
            either an open file object or the filename of a file containing a
            list of connections, in the format required by `FromListConnector`.
        `distributed`:
            if this is True, then each node will read connections from a file
            called `filename.x`, where `x` is the MPI rank. This speeds up
            loading connections for distributed simulations.
        `safe`:
            if True, check that weights and delays have valid values. If False,
            this check is skipped.
        `callback`:
            if True, display a progress bar on the terminal.
    """
    parameter_names = ('filename', 'distributed')

    def __init__(self, file, distributed=False, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe=safe, callback=callback)
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='r')
        self.file = file
        self.distributed = distributed

    def connect(self, projection):
        """Connect-up a Projection."""
        if self.distributed:
            self.file.rename("%s.%d" % (self.file.name,
                                        projection._simulator.state.mpi_rank))
        self.conn_list = self.file.read()
        FromListConnector.connect(self, projection)


class FixedNumberConnector(MapConnector):
    # base class - should not be instantiated
    parameter_names = ('allow_self_connections', 'n')

    def __init__(self, n, allow_self_connections=True,
                 space=Space(), safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, space, safe, callback)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        if isinstance(n, int):
            self.n = n
            assert n >= 0
        elif isinstance(n, RandomDistribution):
            self.rand_distr = n
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) >= 0), "the random distribution produces negative numbers"
        else:
            raise TypeError("n must be an integer or a RandomDistribution object")


class FixedNumberPostConnector(FixedNumberConnector):
    """
    Each pre-synaptic neuron is connected to exactly `n` post-synaptic neurons
    chosen at random.

    If `n` is less than the size of the post-synaptic population, there are no
    multiple connections, i.e., no instances of the same pair of neurons being
    multiply connected. If `n` is greater than the size of the post-synaptic
    population, all possible single connections are made before starting to add
    duplicate connections.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `n`:
            either a positive integer, or a `RandomDistribution` that produces
            positive integers. If `n` is a `RandomDistribution`, then the
            number of post-synaptic neurons is drawn from this distribution
            for each pre-synaptic neuron.
        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
    """

    def connect(self, projection):
        assert projection.rng.parallel_safe
        # this is probably very inefficient, would be better to use
        # divergent connect
        shuffle = numpy.array([projection.rng.permutation(numpy.arange(projection.post.size))
                               for i in range(projection.pre.size)])
        n = self.n
        if hasattr(self, "rand_distr"):
            n = self.rand_distr.next(projection.pre.size)
        f_ij = lambda i,j: shuffle[:, j] < n
        connection_map = LazyArray(f_ij, projection.shape)
        if not self.allow_self_connections and projection.pre == projection.post:
            connection_map *= LazyArray(lambda i,j: i != j, shape=projection.shape)
        self._connect_with_map(projection, connection_map)


class FixedNumberPreConnector(FixedNumberConnector):
    """
    Each post-synaptic neuron is connected to exactly `n` pre-synaptic neurons
    chosen at random.

    If `n` is less than the size of the pre-synaptic population, there are no
    multiple connections, i.e., no instances of the same pair of neurons being
    multiply connected. If `n` is greater than the size of the pre-synaptic
    population, all possible single connections are made before starting to add
    duplicate connections.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `n`:
            either a positive integer, or a `RandomDistribution` that produces
            positive integers. If `n` is a `RandomDistribution`, then the
            number of pre-synaptic neurons is drawn from this distribution
            for each post-synaptic neuron.
        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
    """

    def connect(self, projection):
        assert projection.rng.parallel_safe
        shuffle = projection.rng.permutation(numpy.arange(projection.pre.size))
        n = self.n
        if hasattr(self, "rand_distr"):
            n = self.rand_distr.next(projection.pre.size)
        f_ij = lambda i,j: shuffle[i] < n
        connection_map = LazyArray(f_ij, projection.shape)
        if not self.allow_self_connections and projection.pre == projection.post:
            connection_map *= LazyArray(lambda i,j: i != j, shape=projection.shape)
        self._connect_with_map(projection, connection_map)


class OneToOneConnector(MapConnector):
    """
    Where the pre- and postsynaptic populations have the same size, connect
    cell *i* in the presynaptic population to cell *i* in the postsynaptic
    population for all *i*.

    Takes any of the standard :class:`Connector` optional arguments.
    """
    parameter_names = tuple()

    def connect(self, projection):
        """Connect-up a Projection."""
        connection_map = LazyArray(lambda i,j: i == j, shape=projection.shape)
        self._connect_with_map(projection, connection_map)


class SmallWorldConnector(Connector):
    """
    Connect cells so as to create a small-world network.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `degree`:
            the region length where nodes will be connected locally.
        `rewiring`:
            the probability of rewiring each edge.
        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
        `n_connections`:
            if specified, the number of efferent synaptic connections per neuron.
    """
    parameter_names = ('allow_self_connections', 'degree', 'rewiring', 'n_connections')

    def __init__(self, degree, rewiring, allow_self_connections=True,
                 n_connections=None, space=Space(),
                 safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, space, safe, callback)
        assert 0 <= rewiring <= 1
        assert isinstance(allow_self_connections, bool)
        self.rewiring               = rewiring
        self.d_expression           = "d < %g" % degree
        self.allow_self_connections = allow_self_connections
        self.n_connections          = n_connections

    def connect(self, projection):
        """Connect-up a Projection."""
        raise NotImplementedError


class CSAConnector(Connector):
    """
    Use the Connection Set Algebra (Djurfeldt, 2012) to connect cells.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `cset`:
            a connection set object.
    """
    parameter_names = ('cset',)

    if haveCSA:
        def __init__ (self, cset, safe=True, callback=None):
            """
            """
            Connector.__init__(self, safe=safe, callback=callback)
            self.cset = cset
            if csa.arity(cset) == 0:
                pass
            else:
                assert csa.arity(cset) == 2, 'must specify mask or connection-set with arity 2'
    else:
        def __init__ (self, cset, safe=True, callback=None):
            raise RuntimeError, "CSAConnector not available---couldn't import csa module"

    @staticmethod
    def isConstant(x):
        return isinstance(x, (int, float))

    def connect(self, projection):
        """Connect-up a Projection."""
        if self.delays is None:
            self.delays = projection._simulator.state.min_delay
        # Cut out finite part
        c = csa.cross((0, projection.pre.size-1), (0, projection.post.size-1)) * self.cset  # can't we cut out just the columns we want?

        if csa.arity(self.cset) == 2:
            # Connection-set with arity 2
            for (i, j, weight, delay) in c:
                projection._convergent_connect([projection.pre[i]], projection.post[j], weight, delay)
        elif CSAConnector.isConstant (self.weights) \
             and CSAConnector.isConstant (self.delays):
            # Mask with constant weights and delays
            for (i, j) in c:
                projection._convergent_connect([projection.pre[i]], projection.post[j], self.weights, self.delays)
        else:
            # Mask with weights and/or delays iterable
            weights = self.weights
            if CSAConnector.isConstant(weights):
                weights = repeat(weights)
            delays = self.delays
            if CSAConnector.isConstant(delays):
                delays = repeat(delays)
            for (i, j), weight, delay in zip (c, weights, delays):
                projection._convergent_connect([projection.pre[i]], projection.post[j], weight, delay)
