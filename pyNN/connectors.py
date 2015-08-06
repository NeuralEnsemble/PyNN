"""
Defines a common implementation of the built-in PyNN Connector classes.

Simulator modules may use these directly, or may implement their own versions
for improved performance.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from __future__ import division
from pyNN.random import RandomDistribution, AbstractRNG, NumpyRNG, get_mpi_config
from pyNN.common.populations import is_conductance
from pyNN.core import IndexBasedExpression
from pyNN import errors, descriptions
from pyNN.recording import files
from pyNN.parameters import LazyArray
from pyNN.standardmodels import StandardSynapseType
import numpy
try:
    from itertools import izip
except ImportError:  #python3.x
    izip = zip
try:
    basestring
except NameError:
    basestring = str
from itertools import repeat
import logging
from copy import copy, deepcopy

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


def _get_rng(rng):
    if isinstance(rng, AbstractRNG):
        return rng
    elif rng is None:
        return NumpyRNG(seed=151985012)
    else:
        raise Exception("rng must be either None, or a subclass of pyNN.random.AbstractRNG")


class Connector(object):
    """
    Base class for connectors.

    All connector sub-classes have the following optional keyword arguments:
        `safe`:
            if True, check that weights and delays have valid values. If False,
            this check is skipped.
        `callback`:
            a function that will be called with the fractional progress of the
            connection routine. An example would be `progress_bar.set_level`.
    """

    def __init__(self, safe=True, callback=None):
        """
        docstring needed
        """
        self.safe = safe
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

    def _generate_distance_map(self, projection):
        position_generators = (projection.pre.position_generator,
                               projection.post.position_generator)
        return LazyArray(projection.space.distance_generator(*position_generators),
                         shape=projection.shape)

    def _parameters_from_synapse_type(self, projection, distance_map=None):
        """
        Obtain the parameters to be used for the connections from the projection's `synapse_type`
        attribute. Each parameter value is a `LazyArray`.
        """
        if distance_map is None:
            distance_map = self._generate_distance_map(projection)
        parameter_space = projection.synapse_type.native_parameters
        # TODO: in the documentation, we claim that a parameter value can be
        #       a list or 1D array of the same length as the number of connections.
        #       We do not currently handle this scenario, although it is only
        #       really useful for fixed-number connectors anyway.
        #       Probably the best solution is to remove the parameter at this stage,
        #       then set it after the connections have already been created.
        parameter_space.shape = (projection.pre.size, projection.post.size)
        for name, map in parameter_space.items():
            if callable(map.base_value):
                if isinstance(map.base_value, IndexBasedExpression):
                    # Assumes map is a function of index and hence requires the projection to
                    # determine its value. It and its index function are copied so as to be able
                    # to set the projection without altering the connector, which would perhaps
                    # not be expected from the 'connect' call.
                    new_map = copy(map)
                    new_map.base_value = copy(map.base_value)
                    new_map.base_value.projection = projection
                    parameter_space[name] = new_map
                else:
                    # Assumes map is a function of distance
                    parameter_space[name] = map(distance_map)
        return parameter_space

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
    Abstract base class for Connectors based on connection maps, where a map is a 2D lazy array
    containing either the (boolean) connectivity matrix (aka adjacency matrix, connection set mask, etc.)
    or the values of a synaptic connection parameter.
    """

    def _standard_connect(self, projection, connection_map_generator, distance_map=None):

        column_indices = numpy.arange(projection.post.size)

        if (projection.synapse_type.native_parameters.parallel_safe
            or hasattr(self, "rng") and self.rng.parallel_safe):

            # If any of the synapse parameters are based on parallel-safe random number generators,
            # we need to iterate over all post-synaptic cells, so we can generate then
            # throw away the random numbers for the non-local nodes.
            logger.debug("Parallel-safe iteration.")
            components = (
                column_indices,
                projection.post._mask_local,
                connection_map_generator())
        else:
            # Otherwise, we only need to iterate over local post-synaptic cells.
            mask = projection.post._mask_local
            components = (
                column_indices[mask],
                repeat(True),
                connection_map_generator(mask))

        parameter_space = self._parameters_from_synapse_type(projection, distance_map)

        # Loop over columns of the connection_map array (equivalent to looping over post-synaptic neurons)
        for count, (col, local, source_mask) in enumerate(izip(*components)):
            # `col`: index of the post-synaptic neuron
            # `local`: boolean - does the post-synaptic neuron exist on this MPI node
            # `source_mask` - boolean numpy array, indicating which of the pre-synaptic neurons should be connected to,
            #                 or a single boolean, meaning connect to all/none of the pre-synaptic neurons
            #                 It can also be an array of addresses
            _proceed = False 
            if source_mask is True or source_mask.any():
                _proceed = True
            elif type(source_mask) == numpy.ndarray:
                if source_mask.dtype == bool:
                    if source_mask.any():
                        _proceed = True
                elif len(source_mask) > 0:
                    _proceed = True
            if _proceed:
                # Convert from boolean to integer mask, if necessary
                if source_mask is True:
                    source_mask = numpy.arange(projection.pre.size, dtype=int)
                elif source_mask.dtype == bool:
                    source_mask = source_mask.nonzero()[0]
            
                # Evaluate the lazy arrays containing the synaptic parameters
                connection_parameters = {}
                for name, map in parameter_space.items():
                    if map.is_homogeneous:
                        connection_parameters[name] = map.evaluate(simplify=True)
                    else:
                        connection_parameters[name] = map[source_mask, col]

#                # Check that parameter values are valid
#                if self.safe:
#                    # (might be cheaper to do the weight and delay check before evaluating the larray)
#                    weights = check_weights(weights, projection.synapse_type, is_conductance(projection.post.local_cells[0]))
#                    delays = check_delays(delays,
#                                          projection._simulator.state.min_delay,
#                                          projection._simulator.state.max_delay)
#                    # TODO: add checks for plasticity parameters

                if local:
                    # Connect the neurons
                    #logger.debug("Connecting to %d from %s" % (col, source_mask))
                    projection._convergent_connect(source_mask, col, **connection_parameters)
                    if self.callback:
                        self.callback(count/projection.post.local_size)

    def _connect_with_map(self, projection, connection_map, distance_map=None):
        """
        Create connections according to a connection map.

        Arguments:

            `projection`:
                the `Projection` that is being created.
            `connection_map`:
                a boolean `LazyArray` of the same shape as `projection`, representing the connectivity matrix.
            `distance_map`:
                TODO
        """
        logger.debug("Connecting %s using a connection map" % projection.label)
        self._standard_connect(projection, connection_map.by_column, distance_map)


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

    def __init__(self, allow_self_connections=True, safe=True,
                 callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections

    def connect(self, projection):
        if not self.allow_self_connections and projection.pre == projection.post:
            connection_map = LazyArray(lambda i,j: i != j, shape=projection.shape)
        elif self.allow_self_connections == 'NoMutual' and projection.pre == projection.post:
            connection_map = LazyArray(lambda i,j: i > j, shape=projection.shape)
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
        `rng`:
            an :class:`RNG` instance used to evaluate whether connections exist
    """
    parameter_names = ('allow_self_connections', 'p_connect')

    def __init__(self, p_connect, allow_self_connections=True,
                 rng=None, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert isinstance(allow_self_connections, bool) or allow_self_connections == 'NoMutual'
        self.allow_self_connections = allow_self_connections
        self.p_connect = float(p_connect)
        assert 0 <= self.p_connect
        self.rng = _get_rng(rng)

    def connect(self, projection):
        random_map = LazyArray(RandomDistribution('uniform', (0, 1), rng=self.rng),
                               projection.shape)
        connection_map = random_map < self.p_connect
        if projection.pre == projection.post:
            if not self.allow_self_connections:
                connection_map *= LazyArray(lambda i,j: i != j, shape=projection.shape)
            elif self.allow_self_connections == 'NoMutual':
                connection_map *= LazyArray(lambda i,j: i > j, shape=projection.shape)
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
        `rng`:
            an :class:`RNG` instance used to evaluate whether connections exist
    """
    parameter_names = ('allow_self_connections', 'd_expression')

    def __init__(self, d_expression, allow_self_connections=True,
                 rng=None, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert isinstance(d_expression, str) or callable(d_expression)
        assert isinstance(allow_self_connections, bool) or allow_self_connections == 'NoMutual'
        try:
            if isinstance(d_expression, str):
                d = 0; assert 0 <= eval(d_expression), eval(d_expression)
                d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
        except ZeroDivisionError as err:
            raise ZeroDivisionError("Error in the distance expression %s. %s" % (d_expression, err))
        self.d_expression = d_expression
        self.allow_self_connections = allow_self_connections
        self.distance_function = eval("lambda d: %s" % self.d_expression)
        self.rng = _get_rng(rng)

    def connect(self, projection):
        distance_map = self._generate_distance_map(projection)
        probability_map = self.distance_function(distance_map)
        random_map = LazyArray(RandomDistribution('uniform', (0, 1), rng=self.rng),
                               projection.shape)
        connection_map = random_map < probability_map
        if projection.pre == projection.post:
            if not self.allow_self_connections:
                connection_map *= LazyArray(lambda i,j: i != j, shape=projection.shape)
            elif self.allow_self_connections == 'NoMutual':
                connection_map *= LazyArray(lambda i,j: i > j, shape=projection.shape)
        self._connect_with_map(projection, connection_map, distance_map)


class IndexBasedProbabilityConnector(MapConnector):
    """
    For each pair of pre-post cells, the connection probability depends on an arbitrary functions
    that takes the indices of the pre and post populations.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `index_expression`:
            a function that takes the two cell indices as inputs and calculates the
            probability matrix from it.
        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
        `rng`:
            an :class:`RNG` instance used to evaluate whether connections exist
    """
    parameter_names = ('allow_self_connections', 'index_expression')

    def __init__(self, index_expression, allow_self_connections=True,
                 rng=None, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert callable(index_expression)
        assert isinstance(index_expression, IndexBasedExpression)
        assert isinstance(allow_self_connections, bool) or allow_self_connections == 'NoMutual'
        self.index_expression = index_expression
        self.allow_self_connections = allow_self_connections
        self.rng = _get_rng(rng)

    def connect(self, projection):
        # The index function is copied so as to avoid the connector being altered by the "connect"
        # function, which is probably unexpected behaviour.
        index_expression = copy(self.index_expression)
        index_expression.projection = projection
        probability_map = LazyArray(index_expression, projection.shape)
        random_map = LazyArray(RandomDistribution('uniform', (0, 1), rng=self.rng),
                               projection.shape)
        connection_map = random_map < probability_map
        if projection.pre == projection.post:
            if not self.allow_self_connections:
                connection_map *= LazyArray(lambda i,j: i != j, shape=projection.shape)
            elif self.allow_self_connections == 'NoMutual':
                connection_map *= LazyArray(lambda i,j: i > j, shape=projection.shape)
        self._connect_with_map(projection, connection_map)


class DisplacementDependentProbabilityConnector(IndexBasedProbabilityConnector):

    class DisplacementExpression(IndexBasedExpression):
        """
        A displacement based expression function used to determine the connection probability
        and the value of variable connection parameters of a projection
        """
        def __init__(self, disp_function):
            """
            `disp_function`: a function that takes a 3xN numpy position matrix and maps each row
                             (displacement) to a probability between 0 and 1
            """
            self._disp_function = disp_function

        def __call__(self, i, j):
            disp = (self.projection.post.positions.T[j] - self.projection.pre.positions.T[i]).T
            return self._disp_function(disp)

    def __init__(self, disp_function, allow_self_connections=True,
                 rng=None, safe=True, callback=None):
        super(DisplacementDependentProbabilityConnector, self).__init__(
                self.DisplacementExpression(disp_function),
                allow_self_connections=allow_self_connections, rng=rng, callback=callback)


class FromListConnector(Connector):
    """
    Make connections according to a list.

    Arguments:
        `conn_list`:
            a list of tuples, one tuple for each connection. Each tuple should contain:
            `(pre_idx, post_idx, p1, p2, ..., pn)` where `pre_idx` is the index
            (i.e. order in the Population, not the ID) of the presynaptic
            neuron, `post_idx` is the index of the postsynaptic neuron, and
            p1, p2, etc. are the synaptic parameters (e.g. weight, delay,
            plasticity parameters).
        `column_names`:
            the names of the parameters p1, p2, etc. If not provided, it is
            assumed the parameters are 'weight', 'delay' (for backwards
            compatibility).
        `safe`:
            if True, check that weights and delays have valid values. If False,
            this check is skipped.
        `callback`:
            if True, display a progress bar on the terminal.
    """
    parameter_names = ('conn_list',)

    def __init__(self, conn_list, column_names=None, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe=safe, callback=callback)
        self.conn_list = numpy.array(conn_list)
        if len(conn_list) > 0:
            n_columns = self.conn_list.shape[1]
        if column_names is None:
            if n_columns == 2:
                self.column_names = ()
            elif n_columns == 4:
                self.column_names = ('weight', 'delay')
            else:
                raise TypeError("Argument 'column_names' is required.")
        else:
            self.column_names = column_names
            if n_columns != len(self.column_names) + 2:
                raise ValueError("connection list has %d parameter columns, but %d column names provided." % (
                                 n_columns - 2, len(self.column_names)))

    def connect(self, projection):
        """Connect-up a Projection."""
        logger.debug("conn_list (original) = \n%s", self.conn_list)
        synapse_parameter_names = projection.synapse_type.get_parameter_names()
        for name in self.column_names:
            if name not in synapse_parameter_names:
                raise ValueError("%s is not a valid parameter for %s" % (
                                 name, projection.synapse_type.__class__.__name__))
        if numpy.any(self.conn_list[:, 0] >= projection.pre.size):
            raise errors.ConnectionError("source index out of range")
        # need to do some profiling, to figure out the best way to do this:
        #  - order of sorting/filtering by local
        #  - use numpy.unique, or just do in1d(self.conn_list)?
        idx = numpy.argsort(self.conn_list[:, 1])
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
        logger.debug("conn_list (sorted by target) = \n%s", self.conn_list)
        logger.debug("left = %s", left)
        logger.debug("right = %s", right)

        for tgt, l, r in zip(local_targets, left, right):
            sources = self.conn_list[l:r, 0].astype(numpy.int)
            connection_parameters = deepcopy(projection.synapse_type.parameter_space)
            connection_parameters.shape = (r-l,)
            for col, name in enumerate(self.column_names, 2):
                connection_parameters.update(**{name: self.conn_list[l:r, col]})
            if isinstance(projection.synapse_type, StandardSynapseType):
                connection_parameters = projection.synapse_type.translate(
                                            connection_parameters)
            connection_parameters.evaluate()
            projection._convergent_connect(sources, tgt, **connection_parameters)


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
        self.column_names = self.file.get_metadata().get('columns', ('weight', 'delay'))
        for ignore in "ij":
            if ignore in self.column_names:
                self.column_names.remove(ignore)
        self.conn_list = self.file.read()
        FromListConnector.connect(self, projection)


class FixedNumberConnector(MapConnector):
    # base class - should not be instantiated
    parameter_names = ('allow_self_connections', 'n')

    def __init__(self, n, allow_self_connections=True, with_replacement=False,
                 rng=None, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert isinstance(allow_self_connections, bool) or allow_self_connections == 'NoMutual'
        self.allow_self_connections = allow_self_connections
        self.with_replacement = with_replacement
        self.n = n
        if isinstance(n, int):
            assert n >= 0
        elif isinstance(n, RandomDistribution):
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) >= 0), "the random distribution produces negative numbers"
        else:
            raise TypeError("n must be an integer or a RandomDistribution object")
        self.rng = _get_rng(rng)

    def _rng_uniform_int_exclude(self, n, size, exclude):
        res = self.rng.next(n, 'uniform_int', {"low": 0, "high": size}, mask_local=False)
        logger.debug("RNG0 res=%s" % res)
        idx = numpy.where(res == exclude)[0]
        logger.debug("RNG1 exclude=%d, res=%s idx=%s" % (exclude, res, idx))
        while idx.size > 0:
            redrawn = self.rng.next(idx.size, 'uniform_int', {"low": 0, "high": size}, mask_local=False)
            res[idx] = redrawn
            idx = idx[numpy.where(res == exclude)[0]]
            logger.debug("RNG2 exclude=%d redrawn=%s res=%s idx=%s" % (exclude, redrawn, res, idx))
        return res


class FixedNumberPostConnector(FixedNumberConnector):
    """
    Each pre-synaptic neuron is connected to exactly `n` post-synaptic neurons
    chosen at random.

    The sampling behaviour is controlled by the `with_replacement` argument.

    "With replacement" means that each post-synaptic neuron is chosen from the
    entire population. There is always therefore a possibility of multiple
    connections between a given pair of neurons.

    "Without replacement" means that once a neuron has been selected, it cannot
    be selected again until the entire population has been selected. This means
    that if `n` is less than the size of the post-synaptic population, there
    are no multiple connections. If `n` is greater than the size of the post-
    synaptic population, all possible single connections are made before
    starting to add duplicate connections.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `n`:
            either a positive integer, or a `RandomDistribution` that produces
            positive integers. If `n` is a `RandomDistribution`, then the
            number of post-synaptic neurons is drawn from this distribution
            for each pre-synaptic neuron.
        `with_replacement`:
            if True, the selection of neurons to connect is made from the
            entire population. If False, once a neuron is selected it cannot
            be selected again until the entire population has been connected.
        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
        `rng`:
            an :class:`RNG` instance used to evaluate which potential connections
            are created.
    """

    def _get_num_post(self):
        if isinstance(self.n, int):
            n_post = self.n
        else:
            n_post = self.n.next()
        return n_post

    def connect(self, projection):
        connections = [[] for i in range(projection.post.size)]
        for source_index in range(projection.pre.size):
            n = self._get_num_post()
            if self.with_replacement:
                if not self.allow_self_connections and projection.pre == projection.post:
                    targets = self._rng_uniform_int_exclude(n, projection.post.size, source_index)
                else:
                    targets = self.rng.next(n, 'uniform_int', {"low": 0, "high": projection.post.size}, mask_local=False)
            else:
                all_cells = numpy.arange(projection.post.size)
                if not self.allow_self_connections and projection.pre == projection.post:
                    all_cells = all_cells[all_cells != source_index]
                full_sets = n // all_cells.size
                remainder = n % all_cells.size
                target_sets = []
                if full_sets > 0:
                    target_sets = [all_cells]*full_sets
                if remainder > 0:
                    target_sets.append(self.rng.permutation(all_cells)[:remainder])
                targets = numpy.hstack(target_sets)
            assert targets.size == n
            for target_index in targets:
                connections[target_index].append(source_index)
        def build_source_masks(mask=None):
            if mask is None:
                return [numpy.array(x) for x in connections]
            else:
                return [numpy.array(x) for x in numpy.array(connections)[mask]]
        self._standard_connect(projection, build_source_masks)


class FixedNumberPreConnector(FixedNumberConnector):
    """
    Each post-synaptic neuron is connected to exactly `n` pre-synaptic neurons
    chosen at random.

    The sampling behaviour is controlled by the `with_replacement` argument.

    "With replacement" means that each pre-synaptic neuron is chosen from the
    entire population. There is always therefore a possibility of multiple
    connections between a given pair of neurons.

    "Without replacement" means that once a neuron has been selected, it cannot
    be selected again until the entire population has been selected. This means
    that if `n` is less than the size of the pre-synaptic population, there
    are no multiple connections. If `n` is greater than the size of the pre-
    synaptic population, all possible single connections are made before
    starting to add duplicate connections.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `n`:
            either a positive integer, or a `RandomDistribution` that produces
            positive integers. If `n` is a `RandomDistribution`, then the
            number of pre-synaptic neurons is drawn from this distribution
            for each post-synaptic neuron.
        `with_replacement`:
            if True, the selection of neurons to connect is made from the
            entire population. If False, once a neuron is selected it cannot
            be selected again until the entire population has been connected.
        `allow_self_connections`:
            if the connector is used to connect a Population to itself, this
            flag determines whether a neuron is allowed to connect to itself,
            or only to other neurons in the Population.
        `rng`:
            an :class:`RNG` instance used to evaluate which potential connections
            are created.
    """

    def _get_num_pre(self, size, mask=None):
        if isinstance(self.n, int):
            if mask is None:
                n_pre = repeat(self.n, size)
            else:
                n_pre = repeat(self.n, mask.sum())
        else:
            if mask is None:
                n_pre = self.n.next(size)
            else:
                if self.n.rng.parallel_safe:
                    n_pre = self.n.next(size)[mask]
                else:
                    n_pre = self.n.next(mask.sum())
        return n_pre

    def connect(self, projection):
        if self.with_replacement:
            if self.allow_self_connections or projection.pre != projection.post:
                def build_source_masks(mask=None):
                    n_pre = self._get_num_pre(projection.post.size, mask)
                    for n in n_pre:
                        sources = self.rng.next(n, 'uniform_int', {"low": 0, "high": projection.pre.size}, mask_local=False)
                        assert sources.size == n
                        yield sources
            else:
                def build_source_masks(mask=None):
                    n_pre = self._get_num_pre(projection.post.size, mask)
                    if self.rng.parallel_safe or mask is None:
                        for i, n in enumerate(n_pre):
                            sources = self._rng_uniform_int_exclude(n, projection.pre.size, i)
                            assert sources.size == n
                            yield sources
                    else:
                        # TODO: use mask to obtain indices i
                        raise NotImplementedError("allow_self_connections=False currently requires a parallel safe RNG.")
        else:
            if self.allow_self_connections or projection.pre != projection.post:
                def build_source_masks(mask=None):
                    # where n > projection.pre.size, first all pre-synaptic cells
                    # are connected one or more times, then the remainder
                    # are chosen randomly
                    n_pre = self._get_num_pre(projection.post.size, mask)
                    all_cells = numpy.arange(projection.pre.size)
                    for n in n_pre:
                        full_sets = n // projection.pre.size
                        remainder = n % projection.pre.size
                        source_sets = []
                        if full_sets > 0:
                            source_sets = [all_cells]*full_sets
                        if remainder > 0:
                            source_sets.append(self.rng.permutation(all_cells)[:remainder])
                        sources = numpy.hstack(source_sets)
                        assert sources.size == n
                        yield sources
            else:
                def build_source_masks(mask=None):
                    # where n > projection.pre.size, first all pre-synaptic cells
                    # are connected one or more times, then the remainder
                    # are chosen randomly
                    n_pre = self._get_num_pre(projection.post.size, mask)
                    all_cells = numpy.arange(projection.pre.size)
                    if self.rng.parallel_safe or mask is None:
                        for i, n in enumerate(n_pre):
                            full_sets = n // (projection.pre.size - 1)
                            remainder = n % (projection.pre.size - 1)
                            allowed_cells = all_cells[all_cells != i]
                            source_sets = []
                            if full_sets > 0:
                                source_sets = [allowed_cells]*full_sets
                            if remainder > 0:
                                source_sets.append(self.rng.permutation(allowed_cells)[:remainder])
                            sources = numpy.hstack(source_sets)
                            assert sources.size == n
                            yield sources
                    else:
                        raise NotImplementedError("allow_self_connections=False currently requires a parallel safe RNG.")

        self._standard_connect(projection, build_source_masks)


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
        `rng`:
            an :class:`RNG` instance used to evaluate which connections
            are created.
    """
    parameter_names = ('allow_self_connections', 'degree', 'rewiring', 'n_connections')

    def __init__(self, degree, rewiring, allow_self_connections=True,
                 n_connections=None, rng=None, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert 0 <= rewiring <= 1
        assert isinstance(allow_self_connections, bool) or allow_self_connections == 'NoMutual'
        self.rewiring               = rewiring
        self.d_expression           = "d < %g" % degree
        self.allow_self_connections = allow_self_connections
        self.n_connections          = n_connections
        self.rng = _get_rng(rng)

    def connect(self, projection):
        """Connect-up a Projection."""
        raise NotImplementedError


class CSAConnector(MapConnector):
    """
    Use the Connection Set Algebra (Djurfeldt, 2012) to connect cells.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `cset`:
            a connection set object.
    """
    parameter_names = ('cset',)

    if haveCSA:
        def __init__(self, cset, safe=True, callback=None):
            """
            """
            Connector.__init__(self, safe=safe, callback=callback)
            self.cset = cset
            if csa.arity(cset) == 0:
                pass
            else:
                assert csa.arity(cset) == 2, 'must specify mask or connection-set with arity 2'
    else:
        def __init__(self, cset, safe=True, callback=None):
            raise RuntimeError("CSAConnector not available---couldn't import csa module")

    def connect(self, projection):
        """Connect-up a Projection."""
        # Cut out finite part
        c = csa.cross((0, projection.pre.size-1), (0, projection.post.size-1)) * self.cset  # can't we cut out just the columns we want?

        if csa.arity(self.cset) == 2:
            # Connection-set with arity 2
            for (i, j, weight, delay) in c:
                projection._convergent_connect([projection.pre[i]], projection.post[j], weight, delay)
        elif csa.arity(self.cset) == 0:
            # inefficient implementation as a starting point
            connection_map = numpy.zeros((projection.pre.size, projection.post.size), dtype=bool)
            for addr in c:
                connection_map[addr] = True
            self._connect_with_map(projection, LazyArray(connection_map))
        else:
            raise NotImplementedError


class CloneConnector(MapConnector):
    """
    Connects cells with the same connectivity pattern as a previous projection.
    """
    parameter_names = ('reference_projection',)

    def __init__(self, reference_projection, safe=True, callback=None):
        """
        Create a new CloneConnector.

        `reference_projection` -- the projection to clone the connectivity pattern from
        """
        MapConnector.__init__(self, safe, callback=callback)
        self.reference_projection = reference_projection

    def connect(self, projection):
        if (projection.pre != self.reference_projection.pre or
            projection.post != self.reference_projection.post):
            raise errors.ConnectionError("Pre and post populations must match between reference ({0}"
                                         "  and {1}) and clone projections ({2} and {3}) for "
                                         "CloneConnector"
                                         .format(self.reference_projection.pre,
                                                 self.reference_projection.post,
                                                 projection.pre, projection.post))
        connection_map = LazyArray(~numpy.isnan(self.reference_projection.get(['weight'], 'array',
                                                                              gather='all')[0]))
        self._connect_with_map(projection, connection_map)


class ArrayConnector(MapConnector):
    """
    Provide an explicit boolean connection matrix, with shape (m, n) where m is
    the size of the presynaptic population and n that of the postsynaptic
    population.
    """
    parameter_names = ('array')

    def __init__(self, array, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        self.array = array

    def connect(self, projection):
        connection_map = LazyArray(self.array, projection.shape)
        self._connect_with_map(projection, connection_map)


class FixedTotalNumberConnector(FixedNumberConnector):
    # base class - should not be instantiated
    parameter_names = ('allow_self_connections', 'n')

    def __init__(self, n, allow_self_connections=True, with_replacement=True,
                 rng=None, safe=True, callback=None):
        """
        Create a new connector.
        """
        Connector.__init__(self, safe, callback)
        assert isinstance(allow_self_connections, bool) or allow_self_connections == 'NoMutual'
        self.allow_self_connections = allow_self_connections
        self.with_replacement = with_replacement
        self.n = n
        if isinstance(n, int):
            assert n >= 0
        elif isinstance(n, RandomDistribution):
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) >= 0), "the random distribution produces negative numbers"
        else:
            raise TypeError("n must be an integer or a RandomDistribution object")
        self.rng = _get_rng(rng)


    def connect(self, projection):
        # Determine number of processes and current rank
        rank, num_processes = get_mpi_config()

        # Assume that targets are equally distributed over processes
        targets_per_process = int(len(projection.post)/num_processes)
            
        # Calculate the number of synapses on each process
        bino = RandomDistribution('binomial',[self.n,targets_per_process/len(projection.post)], rng=self.rng)
        num_conns_on_vp = numpy.zeros(num_processes)
        sum_dist = 0
        sum_partitions = 0
        for k in xrange(num_processes) :
            p_local = targets_per_process / ( len(projection.post) - sum_dist)
            bino.parameters['p'] = p_local
            bino.parameters['n'] = self.n - sum_partitions
            num_conns_on_vp[k] = bino.next()
            sum_dist += targets_per_process
            sum_partitions += num_conns_on_vp[k]
	
        # Draw random sources and targets 
        while num_conns_on_vp[rank] > 0 :
            s_index = self.rng.rng.randint(low=0, high=len(projection.pre.all_cells))
            t_index = self.rng.rng.randint(low=0, high=len(projection.post.local_cells))
            t_index = numpy.where(projection.post.all_cells == int(projection.post.local_cells[t_index]))[0][0]

            # Evaluate the lazy arrays containing the synaptic parameters
            parameter_space = self._parameters_from_synapse_type(projection)
            connection_parameters = {}
            for name, map in parameter_space.items():
                if map.is_homogeneous:
                    connection_parameters[name] = map.evaluate(simplify=True)
                else:
                    connection_parameters[name] = map[source_mask, col]
            
            projection._convergent_connect(numpy.array([s_index]),t_index, **connection_parameters)
            num_conns_on_vp[rank] -=1

            
