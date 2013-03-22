# encoding: utf-8
"""
Common implementation of the Projection class.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
import logging
import operator
from pyNN import random, recording, errors, models, core, descriptions
from pyNN.recording import files
from populations import BasePopulation, Assembly, is_conductance

logger = logging.getLogger("PyNN")
deprecated = core.deprecated
DEFAULT_WEIGHT = 0.0

def check_weight(weight, synapse_type, is_conductance):
    if weight is None:
        weight = DEFAULT_WEIGHT
    if core.is_listlike(weight):
        weight = numpy.array(weight)
        nan_filter = (1 - numpy.isnan(weight)).astype(bool)  # weight arrays may contain NaN, which should be ignored
        filtered_weight = weight[nan_filter]
        all_negative = (filtered_weight <= 0).all()
        all_positive = (filtered_weight >= 0).all()
        if not (all_negative or all_positive):
            raise errors.InvalidWeightError("Weights must be either all positive or all negative")
    elif numpy.isreal(weight):
        all_positive = weight >= 0
        all_negative = weight < 0
    else:
        raise errors.InvalidWeightError("Weight must be a number or a list/array of numbers.")
    if is_conductance or synapse_type == 'excitatory':
        if not all_positive:
            raise errors.InvalidWeightError("Weights must be positive for conductance-based and/or excitatory synapses")
    elif is_conductance == False and synapse_type == 'inhibitory':
        if not all_negative:
            raise errors.InvalidWeightError("Weights must be negative for current-based, inhibitory synapses")
    else:  # is_conductance is None. This happens if the cell does not exist on the current node.
        logger.debug("Can't check weight, conductance status unknown.")
    return weight


class Projection(object):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to
    set parameters of those connections, including of plasticity mechanisms.
    """

    def __init__(self, presynaptic_neurons, postsynaptic_neurons, method,
                 source=None, target=None, synapse_dynamics=None,
                 label=None, rng=None):
        """
        presynaptic_neurons and postsynaptic_neurons - Population, PopulationView
                                                       or Assembly objects.

        source - string specifying which attribute of the presynaptic cell
                 signals action potentials. This is only needed for
                 multicompartmental cells with branching axons or
                 dendrodendriticsynapses. All standard cells have a single
                 source, and this is the default.

        target - string specifying which synapse on the postsynaptic cell to
                 connect to. For standard cells, this can be 'excitatory' or
                 'inhibitory'. For non-standard cells, it could be 'NMDA', etc.
                 If target is not given, the default values of 'excitatory' is
                 used.

        method - a Connector object, encapsulating the algorithm to use for
                 connecting the neurons.

        synapse_dynamics - a `standardmodels.SynapseDynamics` object specifying
                 which synaptic plasticity mechanisms to use.

        rng - specify an RNG object to be used by the Connector.
        """
        for prefix, pop in zip(("pre", "post"),
                               (presynaptic_neurons, postsynaptic_neurons)):
            if not isinstance(pop, (BasePopulation, Assembly)):
                raise errors.ConnectionError("%ssynaptic_neurons must be a Population, PopulationView or Assembly, not a %s" % (prefix, type(pop)))
        
        if isinstance(postsynaptic_neurons, Assembly):
            if not postsynaptic_neurons._homogeneous_synapses:
                raise Exception('Projection to an Assembly object can be made only with homogeneous synapses types')
            
        self.pre    = presynaptic_neurons  #  } these really
        self.source = source               #  } should be
        self.post   = postsynaptic_neurons #  } read-only
        self.target = target               #  }
        self.label  = label
        if isinstance(rng, random.AbstractRNG):
            self.rng = rng
        elif rng is None:
            self.rng = random.NumpyRNG(seed=151985012)
        else:
            raise Exception("rng must be either None, or a subclass of pyNN.random.AbstractRNG")
        self._method = method
        self.synapse_dynamics = synapse_dynamics
        #self.connection = None # access individual connections. To be defined by child, simulator-specific classes
        self.weights = []
        if label is None:
            if self.pre.label and self.post.label:
                self.label = "%s→%s" % (self.pre.label, self.post.label)
        if self.synapse_dynamics:
            assert isinstance(self.synapse_dynamics, models.BaseSynapseDynamics), \
              "The synapse_dynamics argument, if specified, must be a models.BaseSynapseDynamics object, not a %s" % type(synapse_dynamics)

    def __len__(self):
        """Return the total number of local connections."""
        raise NotImplementedError

    def size(self, gather=True):
        """
        Return the total number of connections.
            - only local connections, if gather is False,
            - all connections, if gather is True (default)
        """
        if gather:
            n = len(self)
            return recording.mpi_sum(n)
        else:
            return len(self)

    def __repr__(self):
        return 'Projection("%s")' % self.label

    def __getitem__(self, i):
        """Return the `i`th connection within the Projection."""
        raise NotImplementedError

    def __iter__(self):
        """Return an iterator over all connections on the local MPI node."""
        for i in range(len(self)):
            yield self[i]

    # --- Methods for setting connection parameters ---------------------------

    def set(self, name, value):
        """
        Set connection attributes for all connections on the local MPI node.
        
        `name`  -- attribute name
        
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        raise NotImplementedError

    @deprecated("set('weight', w)")
    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the projection, or a 2D array with the same dimensions as the
        connectivity matrix (as returned by `getWeights(format='array')`).
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        # should perhaps add a "distribute" argument, for symmetry with "gather" in getWeights()
        # if post is an Assembly, some components might have cond-synapses, others curr, so need a more sophisticated check here
        w = check_weight(w, self.synapse_type, is_conductance(self.post.local_cells[0]))
        self.set('weight', w)

    @deprecated("set('weight', rand_distr)")
    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        self.set('weight', rand_distr.next(len(self)))

    @deprecated("set('delay', d)")
    def setDelays(self, d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the projection, or a 2D array with the same dimensions as the
        connectivity matrix (as returned by `getDelays(format='array')`).
        """
        self.set('delay', d)

    @deprecated("set('delay', rand_distr)")
    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        self.set('delay', rand_distr.next(len(self)))

    @deprecated("set(param, value)")
    def setSynapseDynamics(self, param, value):
        """
        Set parameters of the dynamic synapses for all connections in this
        projection.
        """
        self.set(param, value)

    @deprecated("set(param, value)")
    def randomizeSynapseDynamics(self, param, rand_distr):
        """
        Set parameters of the synapse dynamics to values taken from rand_distr
        """
        self.set(param, rand_distr.next(len(self)))

    # --- Methods for writing/reading information to/from file. ---------------

    def get(self, parameter_name, format, gather=True):
        """
        Get the values of a given attribute (weight or delay) for all
        connections in this Projection.
        
        `parameter_name` -- name of the attribute whose values are wanted.
        
        `format` -- "list" or "array". Array format implicitly assumes that all
                    connections belong to a single Projection.
        
        Return a list or a 2D Numpy array. The array element X_ij contains the
        attribute value for the connection from the ith neuron in the pre-
        synaptic Population to the jth neuron in the post-synaptic Population,
        if a single such connection exists. If there are no such connections,
        X_ij will be NaN. If there are multiple such connections, the summed
        value will be given, which makes some sense for weights, but is
        pretty meaningless for delays. 
        """
        raise NotImplementedError

    @deprecated("get('weight', format, gather)")
    def getWeights(self, format='list', gather=True):
        """
        Get synaptic weights for all connections in this Projection.

        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D weight array (with NaN for non-existent
        connections). Note that for the array format, if there is more than
        one connection between two cells, the summed weight will be given.
        """
        if gather:
            logger.error("getWeights() with gather=True not yet implemented")
        return self.get('weight', format)

    @deprecated("get('delay', format, gather)")
    def getDelays(self, format='list', gather=True):
        """
        Get synaptic delays for all connections in this Projection.

        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D delay array (with NaN for non-existent
        connections).
        """
        if gather:
            logger.error("getDelays() with gather=True not yet implemented")
        return self.get('delay', format)

    @deprecated("get(parameter_name, format, gather)")
    def getSynapseDynamics(self, parameter_name, format='list', gather=True):
        """
        Get parameters of the dynamic synapses for all connections in this
        Projection.
        """
        if gather:
            logger.error("getSynapseDynamics() with gather=True not yet implemented")
        return self.get(parameter_name, format)

    @deprecated("save('all', file, format, gather)")
    def saveConnections(self, file, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')
        
        lines = []
        if not compatible_output:
            for c in self.connections:
                lines.append([c.source, c.target, c.weight, c.delay])
        else:
            for c in self.connections: 
                lines.append([self.pre.id_to_index(c.source), self.post.id_to_index(c.target), c.weight, c.delay])
        
        if gather == True and self._simulator.state.num_processes > 1:
            all_lines = { self._simulator.state.mpi_rank: lines }
            all_lines = recording.gather_dict(all_lines)
            if self._simulator.state.mpi_rank == 0:
                lines = reduce(operator.add, all_lines.values())
        elif self._simulator.state.num_processes > 1:
            file.rename('%s.%d' % (file.name, self._simulator.state.mpi_rank))
        
        logger.debug("--- Projection[%s].__saveConnections__() ---" % self.label)
        
        if gather == False or self._simulator.state.mpi_rank == 0:
            file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
            file.close()

    @deprecated("save('weight', file, format, gather)")
    def printWeights(self, file, format='list', gather=True):
        """
        Print synaptic weights to file. In the array format, zeros are printed
        for non-existent connections.
        """
        weights = self.get('weight', format=format, gather=gather)
        
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')
        
        if format == 'array':
            weights = numpy.where(numpy.isnan(weights), 0.0, weights)
        file.write(weights, {})
        file.close()    

    @deprecated("save('delay', file, format, gather)")
    def printDelays(self, file, format='list', gather=True):
        """
        Print synaptic weights to file. In the array format, zeros are printed
        for non-existent connections.
        """
        delays = self.getDelays(format=format, gather=gather)
        
        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')
        
        if format == 'array':
            delays = numpy.where(numpy.isnan(delays), 0.0, delays)
        file.write(delays, {})
        file.close()  

    @deprecated("numpy.histogram()")
    def weightHistogram(self, min=None, max=None, nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        weights = self.getWeights(format='list', gather=True)
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
            "target": self.target,
            "size_local": len(self),
            "size": self.size(gather=True),
            "connector": self._method.describe(template=None),
            "plasticity": None,
        }
        if self.synapse_dynamics:
            context.update(plasticity=self.synapse_dynamics.describe(template=None))
        return descriptions.render(engine, template, context)
