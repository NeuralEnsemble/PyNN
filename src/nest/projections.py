# -*- coding: utf-8 -*-
"""
NEST v2 implementation of the PyNN API.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id: __init__.py 1216 2012-09-11 13:57:27Z apdavison $
"""

import numpy
import nest
import logging
from itertools import repeat
from pyNN import common, errors, core
from pyNN.random import RandomDistribution
from . import simulator
from .synapses import NativeSynapseDynamics, NativeSynapseMechanism


logger = logging.getLogger("PyNN")


class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population,
                 method, source=None,
                 target=None, synapse_dynamics=None, label=None, rng=None):
        __doc__ = common.Projection.__init__.__doc__
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target,
                                   synapse_dynamics, label, rng)
        if self.synapse_dynamics:
            synapse_dynamics = self.synapse_dynamics
            self.synapse_dynamics._set_tau_minus(self.post.local_cells)
        else:
            synapse_dynamics = NativeSynapseDynamics("static_synapse")
        self.synapse_model = synapse_dynamics._get_nest_synapse_model("projection_%d" % Projection._nProj)
        self._sources = []
        self._connections = None

        # Create connections
        method.connect(self)

    def __getitem__(self, i):
        """Return the `i`th connection on the local MPI node."""
        if isinstance(i, int):
            if i < len(self):
                return simulator.Connection(self, i)
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [simulator.Connection(self, j) for j in range(i.start, i.stop, i.step or 1)]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))

    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return nest.GetDefaults(self.synapse_model)['num_connections']

    @property
    def connections(self):
        if self._connections is None:
            self._sources = numpy.unique(self._sources)
            self._connections = nest.FindConnections(self._sources, synapse_type=self.synapse_model)
        return self._connections

    def _set_tsodyks_params(self):
        if 'tsodyks' in self.synapse_model: # there should be a better way to do this. In particular, if the synaptic time constant is changed
                                            # after creating the Projection, tau_psc ought to be changed as well.
            assert self.synapse_type in ('excitatory', 'inhibitory'), "only basic synapse types support Tsodyks-Markram connections"
            logger.debug("setting tau_psc")
            targets = nest.GetStatus(self.connections, 'target')
            if self.synapse_type == 'inhibitory':
                param_name = self.post.local_cells[0].celltype.translations['tau_syn_I']['translated_name']
            if self.synapse_type == 'excitatory':
                param_name = self.post.local_cells[0].celltype.translations['tau_syn_E']['translated_name']
            tau_syn = nest.GetStatus(targets, (param_name))
            nest.SetStatus(self.connections, 'tau_psc', tau_syn)

    def _divergent_connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons.

        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        # are we sure the targets are all on the current node?
        if core.is_listlike(source):
            assert len(source) == 1 
            source = source[0]
        if not core.is_listlike(targets):
            targets = [targets]
        assert len(targets) > 0

        if self.synapse_type not in targets[0].celltype.synapse_types:
            raise errors.ConnectionError("User gave synapse_type=%s, synapse_type must be one of: %s" % ( self.synapse_type, "'"+"', '".join(st for st in targets[0].celltype.synapse_types or ['*No connections supported*']))+"'" )
        weights = numpy.array(weights)*1000.0 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                                 # Using convention in this way is not ideal. We should
                                 # be able to look up the units used by each model somewhere.

        # currently we cannot handle the case where `post` is an Assembly and all the synapses are not of the same type
        # so we raise an Exception
        if isinstance(self.post, Assembly) and not self.post._homogeneous_synapses:
            raise errors.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s" % ("Cannot handle Assemblies with non-homogeneous synapse type", source, targets, weights, delays))

        if self.synapse_type == 'inhibitory' and self.post.conductance_based:
            weights *= -1 # NEST wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        elif isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        elif isinstance(delays, float):
            delays = [delays]

        if targets[0].celltype.standard_receptor_type:
            try:
                nest.DivergentConnect([source], targets, weights, delays, self.synapse_model)
            except nest.NESTError, e:
                raise errors.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                             e, source, targets, weights, delays, self.synapse_model))
        else:
            for target, w, d in zip(targets, weights, delays):
                nest.Connect([source], [target], {'weight': w, 'delay': d, 'receptor_type': target.celltype.get_receptor_type(self.synapse_type)})
        self._connections = None # reset the caching of the connection list, since this will have to be recalculated
        self._sources.append(source)

    def _convergent_connect(self, sources, target, weights, delays, **plasticity_attributes):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `sources`  -- a 1D array of pre-synaptic cell IDs
        `target`   -- the ID of the post-synaptic cell.
        `weight`   -- a 1D array of connection weights, of the same length as
                      `sources`, or a single weight value.
        `delays`   -- a 1D array of connection delays, of the same length as
                      `sources`, or a single delay value.
        """
        assert len(sources) > 0, sources
        if self.synapse_type not in target.celltype.synapse_types:
            assert len(target.celltype.synapse_types) > 0
            valid_types = "', '".join(target.celltype.synapse_types)
            raise errors.ConnectionError("User gave synapse_type=%s, synapse_type must be one of: '%s'" % (self.synapse_type,  valid_types))
        weights *= 1000.0   # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                            # Using convention in this way is not ideal. We should
                            # be able to look up the units used by each model somewhere.
        if self.synapse_type == 'inhibitory' and common.is_conductance(target):
            weights *= -1 # NEST wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        if target.celltype.standard_receptor_type:
            try:
                nest.ConvergentConnect(sources.astype(int).tolist(), [target], weights, delays, self.synapse_model)
            except nest.NESTError, e:
                raise errors.ConnectionError("%s. sources=%s, target=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                             e, sources, target, weights, delays, self.synapse_model))
        else:
            if numpy.isscalar(weights):
                weights = repeat(weights)
            if numpy.isscalar(delays):
                delays = repeat(delays)
            for source, w, d in zip(sources, weights, delays): # need to handle case where weights, delays are floats
                nest.Connect([source], [target], {'weight': w, 'delay': d, 'receptor_type': target.celltype.get_receptor_type(self.synapse_type)})
        self._connections = None # reset the caching of the connection list, since this will have to be recalculated
        self._sources.extend(sources)
        if plasticity_attributes:
            connections = nest.FindConnections(sources.astype(int), int(target), self.synapse_model)
            for name, value in plasticity_attributes.items():
                nest.SetStatus(connections, name, value)

    def set(self, **attributes):
        __doc__ = common.Projection.set.__doc__
        for name, value in attributes.items():
            if numpy.isscalar(value):
                value = float(value)
            elif isinstance(value, numpy.ndarray) and len(value.shape) == 2:
                value_list = []
                connection_parameters = nest.GetStatus(self.connections, ('source', 'target'))
                for conn in connection_parameters:
                    addr = self.pre.id_to_index(conn['source']), self.post.id_to_index(conn['target'])
                    try:
                        val = value[addr]
                    except IndexError, e:
                        raise IndexError("%s. addr=%s" % (e, addr))
                    if numpy.isnan(val):
                        raise Exception("Array contains no value for synapse from %d to %d" % (c.source, c.target))
                    else:
                        value_list.append(val)
                value = value_list
            elif core.is_listlike(value):
                value = numpy.array(value)
            elif isinstance(value, RandomDistribution):
                value = value.next(len(self))
            else:
                raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")

            if name == 'weights':
                value *= 1000.0
                if self.synapse_type == 'inhibitory' and common.is_conductance(self.post[0]):
                    value *= -1 # NEST wants negative values for inhibitory weights, even if these are conductances
                name = "weight"
            elif name == 'delays':
                name = "delay"
            else:
                #translation = self.synapse_dynamics.reverse_translate({name: value})
                #name, value = translation.items()[0]
                translated_name = None
                if self.synapse_dynamics.fast:
                    if name in self.synapse_dynamics.fast.translations:
                        translated_name = self.synapse_dynamics.fast.translations[name]["translated_name"] # a hack
                if translated_name is None:
                    if self.synapse_dynamics.slow:
                        for component_name in "timing_dependence", "weight_dependence", "voltage_dependence":
                            component = getattr(self.synapse_dynamics.slow, component_name)
                            if component and name in component.translations:
                                translated_name = component.translations[name]["translated_name"]
                                break
                if translated_name:
                    name = translated_name
    
            try:
                nest.SetStatus(self.connections, name, value) # perhaps take this out of the loop, and set multiple values at once in a dict?
            except nest.NESTError, e:
                n = 1
                if hasattr(value, '__len__'):
                    n = len(value)
                raise Exception("%s. Trying to set %d values." % (e, n))

    def saveConnections(self, file, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        import operator

        if isinstance(file, basestring):
            file = recording.files.StandardTextFile(file, mode='w')

        lines   = nest.GetStatus(self.connections, ('source', 'target', 'weight', 'delay'))

        if gather == True and num_processes() > 1:
            all_lines = { rank(): lines }
            all_lines = recording.gather_dict(all_lines)
            if rank() == 0:
                lines = reduce(operator.add, all_lines.values())
        elif num_processes() > 1:
            file.rename('%s.%d' % (file.name, rank()))
        logger.debug("--- Projection[%s].__saveConnections__() ---" % self.label)

        if gather == False or rank() == 0:
            lines       = numpy.array(lines, dtype=float)
            lines[:,2] *= 0.001
            if compatible_output:
                lines[:,0] = self.pre.id_to_index(lines[:,0])
                lines[:,1] = self.post.id_to_index(lines[:,1])
            file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
            file.close()

    def _get_attributes_as_list(self, *names):
        nest_names = []
        for name in names:
            if name[-1] == "s":  # weights --> weight, delays --> delay
                nest_names.append(name[:-1])
            else:
                nest_names.append(name)
        values = nest.GetStatus(self.connections, nest_names)
        if 'weights' in names: # other attributes could also have scale factors - need to use translation mechanisms
            values = numpy.array(values) # ought to preserve int type for source, target
            scale_factors = numpy.ones(len(names))
            scale_factors[names.index('weights')] = 0.001
            values *= scale_factors
            values = values.tolist()
        return values        

    def _get_attributes_as_arrays(self, *names):
        all_values = []
        for attribute_name in names:
            if attribute_name[-1] == "s":  # weights --> weight, delays --> delay
                attribute_name = attribute_name[:-1]
            value_arr = numpy.nan * numpy.ones((self.pre.size, self.post.size))
            connection_attributes = nest.GetStatus(self.connections, ('source', 'target', attribute_name))
            for conn in connection_attributes:
                # (offset is always 0,0 for connections created with connect())
                src, tgt, value = conn
                addr = self.pre.id_to_index(src), self.post.id_to_index(tgt)
                if numpy.isnan(value_arr[addr]):
                    value_arr[addr] = value
                else:
                    value_arr[addr] += value
            if attribute_name == 'weight':
                value_arr *= 0.001
                if self.synapse_type == 'inhibitory' and self.post.conductance_based:
                    value_arr *= -1 # NEST uses negative values for inhibitory weights, even if these are conductances
            all_values.append(value_arr)
        return all_values
