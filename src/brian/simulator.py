# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API.

Functions and classes useable by the common implementation:

Functions:
    create_cells()
    reset()
    run()

Classes:
    ID
    Recorder
    ConnectionManager
    Connection

Attributes:
    state -- a singleton instance of the _State class.
    recorder_list

All other functions and classes are private, and should not be used by other
modules.


:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import logging
import brian
import numpy
from itertools import izip
import scipy.sparse
from pyNN import common, errors, core

mV = brian.mV
ms = brian.ms
nA = brian.nA
uS = brian.uS
Hz = brian.Hz

# Global variables
recorder_list = []
ZERO_WEIGHT = 1e-99

logger = logging.getLogger("PyNN")

# --- Internal Brian functionality --------------------------------------------

def _new_property(obj_hierarchy, attr_name, units):
    """
    Return a new property, mapping attr_name to obj_hierarchy.attr_name.

    For example, suppose that an object of class A has an attribute b which
    itself has an attribute c which itself has an attribute d. Then placing
      e = _new_property('b.c', 'd')
    in the class definition of A makes A.e an alias for A.b.c.d
    """
    def set(self, value):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        setattr(obj, attr_name, value*units)
    def get(self):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        return getattr(obj, attr_name)/units
    return property(fset=set, fget=get)

def nesteddictwalk(d):
    """
    Walk a nested dict structure, returning all values in all dicts.
    """
    for value1 in d.values():
        if isinstance(value1, dict):
            for value2 in nesteddictwalk(value1):  # recurse into subdict
                yield value2
        else:
            yield value1


class PlainNeuronGroup(brian.NeuronGroup):

    def __init__(self, n, equations, **kwargs):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        brian.NeuronGroup.__init__(self, n, model=equations,
                                   compile=True,
                                   clock=clock,
                                   max_delay=max_delay,
                                   freeze=True,
                                   **kwargs)
        self.parameter_names = equations._namespace.keys()
        for var in ('v', 'ge', 'gi', 'ie', 'ii'): # can probably get this list from brian
            if var in self.parameter_names:
                self.parameter_names.remove(var)
        self.initial_values = {}
        self._S0 = self._S[:,0]

    def initialize(self):
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)

class ThresholdNeuronGroup(brian.NeuronGroup):

    def __init__(self, n, equations, **kwargs):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        brian.NeuronGroup.__init__(self, n, model=equations,
                                   compile=True,
                                   clock=clock,
                                   max_delay=max_delay,
                                   freeze=True,
                                   **kwargs)
        self.parameter_names = equations._namespace.keys() + ['v_thresh', 'v_reset', 'tau_refrac']
        for var in ('v', 'ge', 'gi', 'ie', 'ii'): # can probably get this list from brian
            if var in self.parameter_names:
                self.parameter_names.remove(var)
        self.initial_values = {}
        self._S0 = self._S[:,0]

    tau_refrac = _new_property('_resetfun', 'period', ms)
    v_reset    = _new_property('_resetfun', 'resetvalue', mV)
    v_thresh   = _new_property('_threshold', 'threshold', mV)

    def initialize(self):
        for variable, values in self.initial_values.items():
            setattr(self, variable, values)

class PoissonGroupWithDelays(brian.PoissonGroup):

    def __init__(self, N, rates=0):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        brian.NeuronGroup.__init__(self, N, model=brian.LazyStateUpdater(),
                                   threshold=brian.PoissonThreshold(),
                                   clock=clock,
                                   max_delay=max_delay)
        self._variable_rate = True
        self.rates          = rates
        self._S0[0]         = self.rates(self.clock.t)
        self.parameter_names = ['rate', 'start', 'duration']

    def initialize(self):
        pass


class MultipleSpikeGeneratorGroupWithDelays(brian.MultipleSpikeGeneratorGroup):

    def __init__(self, spiketimes):
        try:
            clock = state.simclock
            max_delay = state.max_delay*ms
        except Exception:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        thresh = brian.directcontrol.MultipleSpikeGeneratorThreshold(spiketimes)
        brian.NeuronGroup.__init__(self, len(spiketimes),
                                   model=brian.LazyStateUpdater(),
                                   threshold=thresh,
                                   clock=clock,
                                   max_delay=max_delay)
        self.parameter_names = ['spiketimes']

    def _get_spiketimes(self):
        return self._threshold.spiketimes
    def _set_spiketimes(self, spiketimes):
        assert core.is_listlike(spiketimes)
        if len(spiketimes) == 0 or numpy.isscalar(spiketimes[0]):
            spiketimes = [spiketimes for i in xrange(len(self))]
        assert len(spiketimes) == len(self), "spiketimes (length %d) must contain as many iterables as there are cells in the group (%d)." % (len(spiketimes), len(self))
        self._threshold.set_spike_times(spiketimes)
    spiketimes = property(fget=_get_spiketimes, fset=_set_spiketimes)

    def initialize(self):
        pass


# --- For implementation of get_time_step() and similar functions --------------

class _State(object):
    """Represent the simulator state."""

    def __init__(self, timestep, min_delay, max_delay):
        """Initialize the simulator."""
        self.network       = brian.Network()
        self.network.clock = brian.Clock(t=0*ms, dt=timestep*ms)
        self.initialized   = True
        self.num_processes = 1
        self.mpi_rank      = 0
        self.min_delay     = min_delay
        self.max_delay     = max_delay
        self.gid           = 0

    def _get_dt(self):
        return self.simclock.dt/ms

    def _set_dt(self, timestep):
        self.simclock.set_dt(timestep*ms)
    dt = property(fget=_get_dt, fset=_set_dt)

    @property
    def simclock(self):
        if self.network.clock is None:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        return self.network.clock

    @property
    def t(self):
        return self.simclock.t/ms

    def run(self, simtime):
        self.network.run(simtime * ms)

    def add(self, obj):
        self.network.add(obj)

    @property
    def next_id(self):
        res = self.gid
        self.gid += 1
        return res


def reset():
    """Reset the state of the current network to time t = 0."""
    state.network.reinit()
    for group in state.network.groups:
        group.initialize()

# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    gid = 0

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        if n is None:
            n = gid
        int.__init__(n)
        common.IDMixin.__init__(self)

    def get_native_parameters(self):
        """Return a dictionary of parameters for the Brian cell model."""
        params = {}
        assert hasattr(self.parent_group, "parameter_names"), str(self.celltype)
        index = self.parent.id_to_index(self)
        for name in self.parent_group.parameter_names:
            if name in ['v_thresh', 'v_reset', 'tau_refrac']:
                # parameter shared among all cells
                params[name] = float(getattr(self.parent_group, name))
            elif name in ['rate', 'duration', 'start']:
                params[name] = getattr(self.parent_group.rates, name)[index]
            elif name == 'spiketimes':
                params[name] = getattr(self.parent_group,name)[index]
            else:
                # parameter may vary from cell to cell
                try:
                    params[name] = float(getattr(self.parent_group, name)[index])
                except TypeError, errmsg:
                    raise TypeError("%s. celltype=%s, parameter name=%s" % (errmsg, self.celltype, name))
        return params

    def set_native_parameters(self, parameters):
        """Set parameters of the Brian cell model from a dictionary."""
        index = self.parent.id_to_index(self)
        for name, value in parameters.items():
            if name in ['v_thresh', 'v_reset', 'tau_refrac']:
                setattr(self.parent_group, name, value)
                #logger.warning("This parameter cannot be set for individual cells within a Population. Changing the value for all cells in the Population.")
            elif name in ['rate', 'duration', 'start']:
                getattr(self.parent_group.rates, name)[index] = value
            elif name == 'spiketimes':
                all_spiketimes = [st[st>state.t] for st in self.parent_group.spiketimes]
                all_spiketimes[index] = value
                self.parent_group.spiketimes = all_spiketimes
            else:
                setattr(self.parent_group[index], name, value)

    def set_initial_value(self, variable, value):
        if variable is 'v':
            value *= mV
        self.parent_group.initial_values[variable][self.parent.id_to_local_index(self)] = value

    def get_initial_value(self, variable):
        return self.parent_group.initial_values[variable][self.parent.id_to_local_index(self)]


# --- For implementation of create() and Population.__init__() -----------------

class STDP(brian.STDP):
    '''
    See documentation for brian:class:`STDP` for more details. Options hidden here could be used in a more
    general manner. For example, spike interactions (all-to-all, nearest), or the axonal vs. dendritic delays
    because Brian do support those features....
    '''
    def __init__(self, C, taup, taum, Ap, Am, mu_p, mu_m, wmin=0, wmax=None,
                 delay_pre=None, delay_post=None):
        if wmax is None:
            raise AttributeError, "You must specify the maximum synaptic weight"
        wmax  = float(wmax) # removes units
        wmin  = float(wmin)
        Ap   *= wmax   # removes units
        Am   *= wmax   # removes units
        eqs = brian.Equations('''
            dA_pre/dt  = -A_pre/taup  : 1
            dA_post/dt = -A_post/taum : 1''', taup=taup, taum=taum, wmax=wmax, mu_m=mu_m, mu_p=mu_p)
        pre   = 'A_pre += Ap'
        pre  += '\nw += A_post*pow(w/wmax, mu_m)'

        post  = 'A_post += Am'
        post += '\nw += A_pre*pow(1-w/wmax, mu_p)'
        brian.STDP.__init__(self, C, eqs=eqs, pre=pre, post=post, wmin=wmin, wmax=wmax, delay_pre=None, delay_post=None, clock=None)


class SimpleCustomRefractoriness(brian.Refractoriness):

    @brian.check_units(period=brian.second)
    def __init__(self, resetfun, period=5*brian.msecond, state=0):
        self.period = period
        self.resetfun = resetfun
        self.state = state
        self._periods = {} # a dictionary mapping group IDs to periods
        self.statevectors = {}
        self.lastresetvalues = {}

    def __call__(self,P):
        '''
        Clamps state variable at reset value.
        '''
        # if we haven't computed the integer period for this group yet.
        # do so now
        if id(P) in self._periods:
            period = self._periods[id(P)]
        else:
            period = int(self.period/P.clock.dt)+1
            self._periods[id(P)] = period
        V = self.statevectors.get(id(P),None)
        if V is None:
            V = P.state_(self.state)
            self.statevectors[id(P)] = V
        LRV = self.lastresetvalues.get(id(P),None)
        if LRV is None:
            LRV = numpy.zeros(len(V))
            self.lastresetvalues[id(P)] = LRV
        lastspikes = P.LS.lastspikes()
        self.resetfun(P,lastspikes)             # call custom reset function
        LRV[lastspikes] = V[lastspikes]         # store a copy of the custom resetted values
        clampedindices = P.LS[0:period]
        V[clampedindices] = LRV[clampedindices] # clamp at custom resetted values

    def __repr__(self):
        return 'Custom refractory period, '+str(self.period)


class AdaptiveReset(object):

    def __init__(self, Vr= -70.6 * mV, b=0.0805 * nA):
        self.Vr = Vr
        self.b  = b

    def __call__(self, P, spikes):
        P.v[spikes] = self.Vr
        P.w[spikes] += self.b


# --- For implementation of connect() and Connector classes --------------------

class Connection(object):
    """
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, brian_connection, indices, addr):
        """
        Create a new connection.

        `brian_connection` -- a Brian Connection object (may contain
                              many connections).
        `index` -- the index of the current connection within
                   `brian_connection`, i.e. the nth non-zero element
                   in the weight array.
        `indices` -- the mapping of the x, y coordinates of the established
                     connections, stored by the connection_manager handling those
                     connections.
        """
        # the index is the nth non-zero element
        self.bc                  = brian_connection
        self.addr                = int(indices[0]), int(indices[1])
        self.source, self.target = addr

    def _set_weight(self, w):
        w = w or ZERO_WEIGHT
        self.bc[self.addr] = w*self.bc.weight_units

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        return float(self.bc[self.addr]/self.bc.weight_units)

    def _set_delay(self, d):
        self.bc.delay[self.addr] = d*ms

    def _get_delay(self):
        """Synaptic delay in ms."""
        if isinstance(self.bc, brian.DelayConnection):
            return float(self.bc.delay[self.addr]/ms)
        if isinstance(self.bc, brian.Connection):
            return float(self.bc.delay * self.bc.source.clock.dt/ms)

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)


class ConnectionManager(object):
    """
    Manage synaptic connections, providing methods for creating, listing,
    accessing individual connections.
    """

    def __init__(self, synapse_type, synapse_model=None, parent=None):
        """
        Create a new ConnectionManager.

        `synapse_type` -- the 'physiological type' of the synapse, e.g.
                          'excitatory' or 'inhibitory',or any other key in the
                          `synapses` attibute of the celltype class.
        `synapse_model` -- not used. Present for consistency with other simulators.
        `parent` -- the parent `Projection`, if any.
        """
        self.synapse_type      = synapse_type
        self.synapse_model     = synapse_model
        self.parent            = parent
        self.n                 = {}
        self.brian_connections = {}
        self.indices           = {}
        self._populations      = [{}, {}]

    def __getitem__(self, i):
        """Return the `i`th connection as a Connection object."""
        cumsum_idx     = numpy.cumsum(self.n.values())
        if isinstance(i, slice):
            idx  = numpy.searchsorted(cumsum_idx, numpy.arange(*i.indices(i.stop)), 'left')
            keys = [self.keys[j] for j in idx]
        else:
            idx  = numpy.searchsorted(cumsum_idx, i, 'left')
            keys = self.keys[idx]
        global_indices = self._indices
        if isinstance(i, int):
            if i < len(self):
                pad        = i - cumsum_idx[idx]
                local_idx  = self.indices[keys][0][pad], self.indices[keys][1][pad]
                local_addr = global_indices[0][i], global_indices[1][i]
                return Connection(self.brian_connections[keys], local_idx, local_addr)
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                res = []
                for count, j in enumerate(xrange(*i.indices(i.stop))):
                    key = keys[count]
                    pad = j - cumsum_idx[idx[count]]
                    local_idx  = self.indices[key][0][pad], self.indices[key][1][pad]
                    local_addr = global_indices[0][j], global_indices[1][j]
                    res.append(Connection(self.brian_connections[key], local_idx, local_addr))
                return res
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))

    def __len__(self):
        """Return the total number of connections in this manager."""
        result = 0
        for key in self.keys:
          result += self.n[key]
        return result

    def __connection_generator(self):
        """Yield each connection in turn."""
        global_indices = self._indices
        count          = 0
        for key in self.keys:
            bc = self.brian_connections[key]
            for i in xrange(bc.W.getnnz()):
                local_idx  = self.indices[key][0][i], self.indices[key][0][i]
                local_addr = global_indices[0][count], global_indices[1][count]
                yield Connection(bc, self.indices[key])
                count     += 1

    @property
    def keys(self):
        return self.brian_connections.keys()

    def __iter__(self):
        """Return an iterator over all connections in this manager."""
        return self.__connection_generator()

    def _finalize(self):
        for key in self.keys:
            self.indices[key]  = self.brian_connections[key].W.nonzero()
            self.brian_connections[key].compress()

    @property
    def _indices(self):
        sources = numpy.array([], int)
        targets = numpy.array([], int)
        for key in self.keys:
            paddings = self._populations[0][key[0]], self._populations[1][key[1]]
            sources  = numpy.concatenate((sources, self.indices[key][0] + paddings[0]))
            targets  = numpy.concatenate((targets, self.indices[key][1] + paddings[1]))
        return sources.astype(int), targets.astype(int)

    def _get_brian_connection(self, source_group, target_group, synapse_obj, weight_units, homogeneous=False):
        """
        Return the Brian Connection object that connects two NeuronGroups with a
        given synapse model.

        source_group -- presynaptic Brian NeuronGroup.
        target_group -- postsynaptic Brian NeuronGroup
        synapse_obj  -- name of the variable that will be modified by synaptic
                        input.
        weight_units -- Brian Units object: nA for current-based synapses,
                        uS for conductance-based synapses.
        """
        key = (source_group, target_group, synapse_obj)
        if not self.brian_connections.has_key(key):
            assert isinstance(source_group, brian.NeuronGroup)
            assert isinstance(target_group, brian.NeuronGroup), type(target_group)
            assert isinstance(synapse_obj, basestring), "%s (%s)" % (synapse_obj, type(synapse_obj))
            try:
                max_delay = state.max_delay*ms
            except Exception:
                raise Exception("Simulation timestep not yet set. Need to call setup()")
            if not homogeneous:
                self.brian_connections[key] = brian.DelayConnection(source_group,
                                                               target_group,
                                                               synapse_obj,
                                                               max_delay=max_delay)
            else:
                self.brian_connections[key] = brian.Connection(source_group,
                                                          target_group,
                                                          synapse_obj,
                                                          max_delay=state.max_delay*ms)
            self.brian_connections[key].weight_units = weight_units
            state.add(self.brian_connections[key])
            self.n[key] = 0
        return self.brian_connections[key]

    def _detect_parent_groups(self, cells):
        groups = {}
        for index, cell in enumerate(cells):
            group = cell.parent_group
            if not groups.has_key(group):
                groups[group] = [index]
            else:
                groups[group] += [index]
        return groups

    def connect(self, source, targets, weights, delays, homogeneous=False):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        #print "connecting", source, "to", targets, "with weights", weights, "and delays", delays
        if not core.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        if not isinstance(source, common.IDMixin):
            raise errors.ConnectionError("source should be an ID object, actually %s" % type(source))
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise errors.ConnectionError("Invalid target ID: %s" % target)
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets),len(weights),len(delays))
        if common.is_conductance(targets[0]):
            units = uS
        else:
            units = nA
        synapse_type = self.synapse_type or "excitatory"
        try:
            source_group = source.parent_group
        except AttributeError, errmsg:
            raise errors.ConnectionError("%s. Maybe trying to connect from non-existing cell (ID=%s)." % (errmsg, source))
        groups = self._detect_parent_groups(targets) # we assume here all the targets belong to the same NeuronGroup

        weights = numpy.array(weights) * units
        delays  = numpy.array(delays) * ms
        weights[weights == 0] = ZERO_WEIGHT

        for target_group, indices in groups.items():
            synapse_obj = targets[indices[0]].parent.celltype.synapses[synapse_type]
            bc          = self._get_brian_connection(source_group, target_group, synapse_obj, units, homogeneous)
            padding     = (int(source.parent.first_id), int(targets[indices[0]].parent.first_id))
            src         = int(source) - padding[0]
            mytargets   = numpy.array(targets, int)[indices] - padding[1]
            bc.W.rows[src] = mytargets
            bc.W.data[src] = weights[indices]
            if not homogeneous:
                bc.delayvec.rows[src] = mytargets
                bc.delayvec.data[src] = delays[indices]
            else:
                bc.delay = int(delays[0] / bc.source.clock.dt)
            key = (source_group, target_group, synapse_obj)
            self.n[key] += len(mytargets)

            pop_sources = self._populations[0]
            if len(pop_sources) is 0:
                pop_sources[source_group] = 0
            elif not pop_sources.has_key(source_group):
                pop_sources[source_group] = numpy.sum([len(item) for item in pop_sources.keys()])
            pop_targets = self._populations[1]
            if len(pop_targets) is 0:
                pop_targets[target_group] = 0
            elif not pop_targets.has_key(target_group):
                pop_targets[target_group] = numpy.sum([len(item) for item in pop_targets.keys()])


    def get(self, parameter_name, format):
        """
        Get the values of a given attribute (weight or delay) for all
        connections in this manager.

        `parameter_name` -- name of the attribute whose values are wanted.
        `format` -- "list" or "array". Array format implicitly assumes that all
                    connections belong to a single Projection.

        Return a list or a 2D Numpy array. The array element X_ij contains the
        attribute value for the connection from the ith neuron in the pre-
        synaptic Population to the jth neuron in the post-synaptic Population,
        if such a connection exists. If there are no such connections, X_ij will
        be NaN.
        """
        if self.parent is None:
            raise Exception("Only implemented for connections created via a Projection object, not using connect()")
        values = numpy.array([])
        for key in self.keys:
            bc = self.brian_connections[key]
            if parameter_name == "weight":
                values = numpy.concatenate((values, bc.W.alldata / bc.weight_units))
            elif parameter_name == 'delay':
                if isinstance(bc, brian.DelayConnection):
                    values = numpy.concatenate((values, bc.delay.alldata / ms))
                else:
                    data   = bc.delay * bc.source.clock.dt * numpy.ones(bc.W.getnnz()) /ms
                    values = numpy.concatenate((values, data))
            else:
                raise Exception("Getting parameters other than weight and delay not yet supported.")

        if format == 'list':
            values = values.tolist()
        elif format == 'array':
            values_arr = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            sources, targets = self._indices
            values_arr[sources, targets] = values
            values = values_arr
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values

    def set(self, name, value):
        """
        Set connection attributes for all connections in this manager.

        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        if self.parent is None:
            raise Exception("Only implemented for connections created via a Projection object, not using connect()")
        for key in self.keys:
            bc = self.brian_connections[key]
            padding = 0
            if name == 'weight':
                M = bc.W
                units = bc.weight_units
            elif name == 'delay':
                M = bc.delay
                units = ms
            else:
                raise Exception("Setting parameters other than weight and delay not yet supported.")
            value = value*units
            if numpy.isscalar(value):
                if (name == 'weight') or (name == 'delay' and isinstance(bc, brian.DelayConnection)):
                    for row in xrange(M.shape[0]):
                        M.set_row(row, value)
                elif (name == 'delay' and isinstance(bc, brian.Connection)):
                    bc.delay = int(value / bc.source.clock.dt)
                else:
                    raise Exception("Setting a non appropriate parameter")
            elif isinstance(value, numpy.ndarray) and len(value.shape) == 2:
                if (name == 'delay') and not isinstance(bc, brian.DelayConnection):
                    raise Exception("FastConnector have been used, and only fixed homogeneous delays are allowed")
                address_gen = ((i, j) for i,row in enumerate(bc.W.rows) for j in row)
                for (i,j) in address_gen:
                    M[i,j] = value[i,j]
            elif core.is_listlike(value):
                N = M.getnnz()
                assert len(value[padding:padding+N]) == N
                if (name == 'delay') and not isinstance(bc, brian.DelayConnection):
                    raise Exception("FastConnector have been used: only fixed homogeneous delays are allowed")
                M.alldata[:] = value
            else:
                raise Exception("Values must be scalars or lists/arrays")
            padding += M.getnnz()


# --- Initialization, and module attributes ------------------------------------

state = None  # a Singleton, so only a single instance ever exists
#del _State