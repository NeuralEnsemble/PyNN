# encoding: utf-8
"""
Common implementation of the Projection class, to be sub-classed by
backend-specific Projection classes.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    basestring
    reduce
    xrange
except NameError:
    basestring = str
    from functools import reduce
    xrange = range
import numpy
import logging
import operator
from copy import copy
from pyNN import recording, errors, models, core, descriptions
from pyNN.parameters import ParameterSpace, LazyArray
from pyNN.space import Space
from pyNN.standardmodels import StandardSynapseType
from .populations import BasePopulation, Assembly

logger = logging.getLogger("PyNN")
deprecated = core.deprecated


class Projection(object):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to
    set the parameters of those connections, including the parameters of
    plasticity mechanisms.

    Arguments:
        `presynaptic_neurons` and `postsynaptic_neurons`:
            Population, PopulationView or Assembly objects.
        `source`:
            string specifying which attribute of the presynaptic cell signals
            action potentials. This is only needed for multicompartmental cells
            with branching axons or dendrodendritic synapses. All standard cells
            have a single source, and this is the default.
        `receptor_type`:
            string specifying which synaptic receptor_type type on the postsynaptic cell to connect
            to. For standard cells, this can be 'excitatory' or 'inhibitory'.
            For non-standard cells, it could be 'NMDA', etc. If receptor_type is not
            given, the default values of 'excitatory' is used.
        `connector`:
            a Connector object, encapsulating the algorithm to use for
            connecting the neurons.
        `synapse_type`:
            a SynapseType object specifying which synaptic connection
            mechanisms to use.
        `space`:
            TO DOCUMENT
    """
    _nProj = 0

    def __init__(self, presynaptic_neurons, postsynaptic_neurons, connector,
                 synapse_type=None, source=None, receptor_type=None,
                 space=Space(), label=None):
        """
        Create a new projection, connecting the pre- and post-synaptic neurons.
        """
        for prefix, pop in zip(("pre", "post"),
                               (presynaptic_neurons, postsynaptic_neurons)):
            if not isinstance(pop, (BasePopulation, Assembly)):
                raise errors.ConnectionError("%ssynaptic_neurons must be a Population, PopulationView or Assembly, not a %s" % (prefix, type(pop)))

        if isinstance(postsynaptic_neurons, Assembly):
            if not postsynaptic_neurons._homogeneous_synapses:
                raise errors.ConnectionError('Projection to an Assembly object can be made only with homogeneous synapses types')

        self.pre    = presynaptic_neurons  #  } these really
        self.source = source               #  } should be
        self.post   = postsynaptic_neurons #  } read-only
        self.receptor_type = receptor_type or 'excitatory'  # TO FIX: if weights are negative, default should be 'inhibitory'
        if self.receptor_type not in postsynaptic_neurons.receptor_types:
            valid_types = postsynaptic_neurons.receptor_types
            assert len(valid_types) > 0
            errmsg = "User gave synapse_type=%s, synapse_type must be one of: '%s'"
            raise errors.ConnectionError(errmsg % (self.receptor_type, "', '".join(valid_types)))
        self.label = label
        self.space = space
        self._connector = connector
        self.synapse_type = synapse_type or self._static_synapse_class()
        assert isinstance(self.synapse_type, models.BaseSynapseType), \
              "The synapse_type argument must be a models.BaseSynapseType object, not a %s" % type(synapse_type)
        if label is None:
            if self.pre.label and self.post.label:
                self.label = u"%s→%s" % (self.pre.label, self.post.label)
        Projection._nProj += 1

    def __len__(self):
        """Return the total number of local connections."""
        raise NotImplementedError

    def size(self, gather=True):
        """
        Return the total number of connections.
            - only local connections, if gather is False,
            - all connections, if gather is True (default)
        """
        if gather and self._simulator.state.num_processes > 1:
            n = len(self)
            return recording.mpi_sum(n)
        else:
            return len(self)

    @property
    def shape(self):
        return (self.pre.size, self.post.size)

    def __repr__(self):
        return 'Projection("%s")' % self.label

    def __getitem__(self, i):
        """Return the *i*th connection within the Projection."""
        raise NotImplementedError

    def __iter__(self):
        """Return an iterator over all connections on the local MPI node."""
        for i in range(len(self)):
            yield self[i]

    # --- Methods for setting connection parameters ---------------------------

    def set(self, **attributes):
        """
        Set connection attributes for all connections on the local MPI node.

        Attribute names may be 'weight', 'delay', or the name of any parameter
        of a synapse dynamics model (e.g. 'U' for TsodyksMarkramSynapse).

        Each attribute value may be:
            (1) a single number
            (2) a RandomDistribution object
            (3) a 2D array with the same dimensions as the connectivity matrix
                (as returned by `get(format='array')`
            (4) a mapping function, which accepts a single float argument (the
                distance between pre- and post-synaptic cells) and returns a single value.

        Weights should be in nA for current-based and µS for conductance-based
        synapses. Delays should be in milliseconds.

        Note that where a projection contains multiple connections between a given pair
        of neurons, all these connections will be set to the same value.
        """
        # should perhaps add a "distribute" argument, for symmetry with "gather" in get()

        # Note: we have removed the option:
        #      "a list/1D array of the same length as the number of local connections"
        # because it was proving tricky to implement and was holding up the release.
        # The plan is to add this option back at a later date.

        attributes = self._value_list_to_array(attributes)
        parameter_space = ParameterSpace(attributes,
                                         self.synapse_type.get_schema(),
                                         (self.pre.size, self.post.size))
        parameter_space = self._handle_distance_expressions(parameter_space)
        if isinstance(self.synapse_type, StandardSynapseType):
            parameter_space = self.synapse_type.translate(parameter_space)
        self._set_attributes(parameter_space)

    def _value_list_to_array(self, attributes):
        """Convert a list of connection parameters/attributes to a 2D array."""
        connection_mask = ~numpy.isnan(self.get('weight', format='array', gather='all'))
        for name, value in attributes.items():
            if isinstance(value, list) or (isinstance(value, numpy.ndarray) and value.ndim == 1):
                array_value = numpy.nan * numpy.ones(self.shape)
                array_value[connection_mask] = value
                attributes[name] = array_value
        return attributes

    def _handle_distance_expressions(self, parameter_space):
        # also index-based expressions
        for name, map in parameter_space.items():
            if callable(map.base_value):
                if isinstance(map.base_value, core.IndexBasedExpression):
                    map.base_value.projection = self
                    parameter_space[name] = map
                else:
                    # Assumes map is a function of distance
                    position_generators = (self.pre.position_generator,
                                           self.post.position_generator)
                    distance_map = LazyArray(self.space.distance_generator(*position_generators),
                                             shape=self.shape)
                    parameter_space[name] = map(distance_map)
        return parameter_space

    @deprecated("set(weight=w)")
    def setWeights(self, w):
        self.set(weight=w)

    @deprecated("set(weight=rand_distr)")
    def randomizeWeights(self, rand_distr):
        self.set(weight=rand_distr)

    @deprecated("set(delay=d)")
    def setDelays(self, d):
        self.set(delay=d)

    @deprecated("set(delay=rand_distr)")
    def randomizeDelays(self, rand_distr):
        self.set(delay=rand_distr)

    @deprecated("set(parameter_name=value)")
    def setSynapseDynamics(self, parameter_name, value):
        self.set(parameter_name=value)

    @deprecated("set(name=rand_distr)")
    def randomizeSynapseDynamics(self, parameter_name, rand_distr):
        self.set(parameter_name=rand_distr)

    # --- Methods for writing/reading information to/from file. ---------------

    def get(self, attribute_names, format, gather=True, with_address=True):
        """
        Get the values of a given attribute (weight or delay) for all
        connections in this Projection.

        `attribute_names`:
            name of the attributes whose values are wanted, or a list of such
            names.
        `format`:
            "list" or "array".

        With list format, returns a list of tuples. Each tuple contains the
        indices of the pre- and post-synaptic cell followed by the attribute
        values in the order given in `attribute_names`. Example::

            >>> prj.get(["weight", "delay"], format="list")[:5]
            [(TODO)]

        With array format, returns a tuple of 2D NumPy arrays, one for each
        name in `attribute_names`. The array element X_ij contains the
        attribute value for the connection from the ith neuron in the pre-
        synaptic Population to the jth neuron in the post-synaptic Population,
        if a single such connection exists. If there are no such connections,
        X_ij will be NaN. If there are multiple such connections, the summed
        value will be given, which makes some sense for weights, but is
        pretty meaningless for delays. Example::

            >>> weights, delays = prj.get(["weight", "delay"], format="array")
            >>> weights.shape
            TODO

        TODO: document "with_address"

        Values will be expressed in the standard PyNN units (i.e. millivolts,
        nanoamps, milliseconds, microsiemens, nanofarads, event per second).
        """
        if isinstance(attribute_names, basestring):
            attribute_names = (attribute_names,)
            return_single = True
        else:
            return_single = False
        if isinstance(self.synapse_type, StandardSynapseType):
            attribute_names = self.synapse_type.get_native_names(*attribute_names)
        if format == 'list':
            names = list(attribute_names)
            if with_address:
                names = ["presynaptic_index", "postsynaptic_index"] + names
            values = self._get_attributes_as_list(*names)
            if gather and self._simulator.state.num_processes > 1:
                all_values = { self._simulator.state.mpi_rank: values }
                all_values = recording.gather_dict(all_values, all=(gather=='all'))
                if gather == 'all' or self._simulator.state.mpi_rank == 0:
                    values = reduce(operator.add, all_values.values())
            if not with_address and return_single:
                values = [val[0] for val in values]
            return values
        elif format == 'array':
            if gather and self._simulator.state.num_processes > 1:
                # Node 0 is the only one creating a full connection matrix, and returning it (saving memory)
                # Slaves nodes are returning list of connections, so this may be inconsistent...
                names      = list(attribute_names)
                names      = ["presynaptic_index", "postsynaptic_index"] + names
                values     = self._get_attributes_as_list(*names)
                all_values = { self._simulator.state.mpi_rank: values }
                all_values = recording.gather_dict(all_values, all=(gather=='all'))
                if gather == 'all' or self._simulator.state.mpi_rank == 0:
                    tmp_values = reduce(operator.add, all_values.values())
                    values     = self._get_attributes_as_arrays(*attribute_names)
                    tmp_values = numpy.array(tmp_values)
                    for i in xrange(len(values)):
                        values[i][tmp_values[:, 0].astype(int), tmp_values[:, 1].astype(int)] = tmp_values[:, 2+i]
            else:
                values = self._get_attributes_as_arrays(*attribute_names)
            if return_single:
                if gather == 'all' or self._simulator.state.mpi_rank == 0:
                    assert len(values) == 1, values
                return values[0]
            else:
                return values
        else:
            raise Exception("format must be 'list' or 'array'")

    def _get_attributes_as_list(self, *names):
        return [c.as_tuple(*names) for c in self.connections]

    def _get_attributes_as_arrays(self, *names):
        all_values = []
        for attribute_name in names:
            values = numpy.nan * numpy.ones((self.pre.size, self.post.size))
            if attribute_name[-1] == "s":  # weights --> weight, delays --> delay
                attribute_name = attribute_name[:-1]
            for c in self.connections:
                value = getattr(c, attribute_name)
                addr = (c.presynaptic_index, c.postsynaptic_index)
                if numpy.isnan(values[addr]):
                    values[addr] = value
                else:
                    values[addr] += value   # addition is only appropriate for certain variables
                                            # e.g. weight. Not appropriate for delays.
                                            # What about synaptic parameters, e.g. wmax?
            all_values.append(values)
        return all_values

    @deprecated("get('weight', format, gather)")
    def getWeights(self, format='list', gather=True):
        return self.get('weight', format, gather, with_address=False)

    @deprecated("get('delay', format, gather)")
    def getDelays(self, format='list', gather=True):
        return self.get('delay', format, gather, with_address=False)

    @deprecated("get(parameter_name, format, gather)")
    def getSynapseDynamics(self, parameter_name, format='list', gather=True):
        return self.get(parameter_name, format, gather, with_address=False)

    def save(self, attribute_names, file, format='list', gather=True, with_address=True):
        """
        Print synaptic attributes (weights, delays, etc.) to file. In the array
        format, zeros are printed for non-existent connections.

        Values will be expressed in the standard PyNN units (i.e. millivolts,
        nanoamps, milliseconds, microsiemens, nanofarads, event per second).
        """
        if attribute_names in ('all', 'connections'):
            attribute_names = self.synapse_type.get_parameter_names()
        if isinstance(file, basestring):
            file = recording.files.StandardTextFile(file, mode='wb')
        all_values = self.get(attribute_names, format=format, gather=gather, with_address=with_address)
        if format == 'array':
            all_values = [numpy.where(numpy.isnan(values), 0.0, values)
                          for values in all_values]
        if self._simulator.state.mpi_rank == 0:
            metadata = {"columns": attribute_names}
            if with_address:
                metadata["columns"] = ["i", "j"] + list(metadata["columns"])
            file.write(all_values, metadata)
            file.close()

    @deprecated("save('all', file, format='list', gather=gather)")
    def saveConnections(self, file, gather=True, compatible_output=True):
        self.save('all', file, format='list', gather=gather)

    @deprecated("save('weight', file, format, gather)")
    def printWeights(self, file, format='list', gather=True):
        self.save('weight', file, format, gather)

    @deprecated("save('delay', file, format, gather)")
    def printDelays(self, file, format='list', gather=True):
        """
        Print synaptic weights to file. In the array format, zeros are printed
        for non-existent connections.
        """
        self.save('delay', file, format, gather)

    @deprecated("numpy.histogram()")
    def weightHistogram(self, min=None, max=None, nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        weights = numpy.array(self.get('weight', format='list', gather=True, with_address=False))
        if min is None:
            min = weights.min()
        if max is None:
            max = weights.max()
        bins = numpy.linspace(min, max, nbins+1)
        return numpy.histogram(weights, bins)  # returns n, bins

    def describe(self, template='projection_default.txt', engine='default'):
        """
        Returns a human-readable description of the projection.

        The output may be customized by specifying a different template
        togther with an associated template engine (see ``pyNN.descriptions``).

        If template is None, then a dictionary containing the template context
        will be returned.
        """
        context = {
            "label": self.label,
            "pre": self.pre.describe(template=None),
            "post": self.post.describe(template=None),
            "source": self.source,
            "receptor_type": self.receptor_type,
            "size_local": len(self),
            "size": self.size(gather=True),
            "connector": self._connector.describe(template=None),
            "plasticity": None,
        }
        if self.synapse_type:
            context.update(plasticity=self.synapse_type.describe(template=None))
        return descriptions.render(engine, template, context)


class Connection(object):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """
    pass
