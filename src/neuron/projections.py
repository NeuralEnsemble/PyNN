# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import numpy
import logging
from itertools import izip, repeat
from pyNN import common, errors, core
from pyNN.random import RandomDistribution, NativeRNG
from . import simulator

logger = logging.getLogger("PyNN")

class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population, method,
                 source=None, target=None,
                 synapse_dynamics=None, label=None, rng=None):
        __doc__ = common.Projection.__init__.__doc__
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population, method,
                                   source, target, synapse_dynamics, label, rng)

        ## Deal with short-term synaptic plasticity
        if self.synapse_dynamics and self.synapse_dynamics.fast:
            # need to check it is actually the Ts-M model, even though that is the only one at present!
            parameter_space = self.synapse_dynamics.fast.translated_parameters
            parameter_space.evaluate(mask=self.post._mask_local)
            for cell, P in zip(self.post, parameter_space):
                cell._cell.set_Tsodyks_Markram_synapses(self.synapse_type,
                                                        P['U'], P['tau_rec'],
                                                        P['tau_facil'], P['u0'])
            self.synapse_model = 'Tsodyks-Markram'
        else:
            self.synapse_model = None
        self.connections = []

        ## Create connections
        method.connect(self)

        logger.info("--- Projection[%s].__init__() ---" %self.label)

        ## Deal with long-term synaptic plasticity
        if self.synapse_dynamics and self.synapse_dynamics.slow:
            ddf = self.synapse_dynamics.slow.dendritic_delay_fraction
            if ddf > 0.5 and self._simulator.state.num_processes > 1:
                # depending on delays, can run into problems with the delay from the
                # pre-synaptic neuron to the weight-adjuster mechanism being zero.
                # The best (only?) solution would be to create connections on the
                # node with the pre-synaptic neurons for ddf>0.5 and on the node
                # with the post-synaptic neuron (as is done now) for ddf<0.5
                raise NotImplementedError("STDP with dendritic_delay_fraction > 0.5 is not yet supported for parallel computation.")
            stdp_parameters = self.synapse_dynamics.slow.all_parameters
            stdp_parameters['allow_update_on_post'] = int(False) # for compatibility with NEST
            long_term_plasticity_mechanism = self.synapse_dynamics.slow.possible_models
            for c in self.connections:
                c.useSTDP(long_term_plasticity_mechanism, stdp_parameters, ddf)

        # Check none of the delays are out of bounds. This should be redundant,
        # as this should already have been done in the Connector object, so
        # we could probably remove it.
        delays = [c.nc.delay for c in self.connections]
        if delays:
            assert min(delays) >= self._simulator.state.min_delay

    def __getitem__(self, i):
        __doc__ = common.Projection.__getitem__.__doc__
        if isinstance(i, int):
            if i < len(self):
                return self.connections[i]
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [self.connections[j] for j in range(*i.indices(i.stop))]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))

    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return len(self.connections)

    def _resolve_synapse_type(self):
        if self.synapse_type is None:
            self.synapse_type = weight>=0 and 'excitatory' or 'inhibitory'
        if self.synapse_model == 'Tsodyks-Markram' and 'TM' not in self.synapse_type:
            self.synapse_type += '_TM'

    def _divergent_connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        if not isinstance(source, int) or source > simulator.state.gid_counter or source < 0:
            errmsg = "Invalid source ID: %s (gid_counter=%d)" % (source, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)
        if not core.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise errors.ConnectionError("Invalid target ID: %s" % target)

        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets), len(weights), len(delays))
        self._resolve_synapse_type()
        for target, weight, delay in zip(targets, weights, delays):
            if target.local:
                if "." in self.synapse_type:
                    section, synapse_type = self.synapse_type.split(".")
                    synapse_object = getattr(getattr(target._cell, section), synapse_type)
                else:
                    synapse_object = getattr(target._cell, self.synapse_type)
                nc = simulator.state.parallel_context.gid_connect(int(source), synapse_object)
                nc.weight[0] = weight

                # if we have a mechanism (e.g. from 9ML) that includes multiple
                # synaptic channels, need to set nc.weight[1] here
                if nc.wcnt() > 1 and hasattr(target._cell, "type"):
                    nc.weight[1] = target._cell.type.synapse_types.index(self.synapse_type)
                nc.delay  = delay
                # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested
                self.connections.append(simulator.Connection(source, target, nc))

    def _convergent_connect(self, sources, target, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `sources`  -- a 1D array of pre-synaptic cell IDs
        `target`   -- the ID of the post-synaptic cell.
        `weight`   -- a 1D array of connection weights, of the same length as
                      `sources`, or a single weight value.
        `delays`   -- a 1D array of connection delays, of the same length as
                      `sources`, or a single delay value.
        """
        if not isinstance(target, int) or target > simulator.state.gid_counter or target < 0:
            errmsg = "Invalid target ID: %s (gid_counter=%d)" % (target, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)

        if isinstance(weights, float):
            weights = repeat(weights)
        else:
            assert len(sources) == len(weights)
        if isinstance(delays, float):
            delays = repeat(delays)
        else:
            assert len(sources) == len(delays)

        if self.synapse_type is None:
            self.synapse_type = weight >= 0 and 'excitatory' or 'inhibitory'
        if self.synapse_model == 'Tsodyks-Markram' and 'TM' not in self.synapse_type:
            self.synapse_type += '_TM'
        if target.local:  # can perhaps assert target.local ?
            if "." in self.synapse_type:
                section, synapse_type = self.synapse_type.split(".")
                synapse_object = getattr(getattr(target._cell, section), synapse_type)
            else:
                synapse_object = getattr(target._cell, self.synapse_type)
            for source, weight, delay in izip(sources, weights, delays):
                #logging.debug("Connecting neuron #%s to neuron #%s with synapse type %s, weight %g, delay %g", source, target, self.synapse_type, weight, delay)
                if not isinstance(source, common.IDMixin):
                    raise errors.ConnectionError("Invalid source ID: %s" % source)
                nc = simulator.state.parallel_context.gid_connect(int(source), synapse_object)
                nc.weight[0] = weight
                # if we have a mechanism (e.g. from 9ML) that includes multiple
                # synaptic channels, need to set nc.weight[1] here
                if nc.wcnt() > 1 and hasattr(target._cell, "type"):
                    nc.weight[1] = target._cell.type.synapse_types.index(self.synapse_type)
                nc.delay  = delay
                # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested
                self.connections.append(simulator.Connection(source, target, nc))


    # --- Methods for setting connection parameters ----------------------------

    def set(self, **attributes):
        __doc__ = common.Projection.set.__doc__
        for name, value in attributes.items():
            if numpy.isscalar(value):
                for c in self:
                    setattr(c, name, value)
            elif isinstance(value, numpy.ndarray) and len(value.shape) == 2:
                for c in self.connections:
                    addr = (self.pre.id_to_index(c.source), self.post.id_to_index(c.target))
                    try:
                        val = value[addr]
                    except IndexError, e:
                        raise IndexError("%s. addr=%s" % (e, addr))
                    if numpy.isnan(val):
                        raise Exception("Array contains no value for synapse from %d to %d" % (c.source, c.target))
                    else:
                        setattr(c, name, val)
            elif core.is_listlike(value):
                for c,val in zip(self.connections, value):
                    setattr(c, name, val)
            elif isinstance(value, RandomDistribution):
                if isinstance(value.rng, NativeRNG):
                    rarr = simulator.nativeRNG_pick(len(self),
                                                    value.rng,
                                                    value.name,
                                                    value.parameters)
                else:
                    rarr = value.next(len(self))
                for c,val in zip(self.connections, rarr):
                    setattr(c, name, val)
            else:
                raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")
