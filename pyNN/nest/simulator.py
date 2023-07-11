# -*- coding: utf-8 -*-
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the NEST simulator.

Classes and attributes usable by the common implementation:

Classes:
    ID
    Connection

Attributes:
    state -- a singleton instance of the _State class.

All other functions and classes are private, and should not be used by other
modules.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import os.path
import logging
import tempfile
import warnings
import numpy as np

import nest

from .. import common
from ..core import reraise, find, run_command

logger = logging.getLogger("PyNN")
name = "NEST"  # for use in annotating output data

# The following constants contain the names of NEST model parameters that
# relate to simulation time and so may need to be adjusted by a time offset.
# TODO: Currently contains only parameters that occur in PyNN standard models.
#       We should add parameters from all models that are distributed with NEST
#       in case they are used with PyNN as "native" models.
NEST_VARIABLES_TIME_DIMENSION = ("start", "stop")
NEST_ARRAY_VARIABLES_TIME_DIMENSION = ("spike_times", "amplitude_times", "rate_times")


# --- Building extensions ------------------------------------------------------

def build_extensions(build_dir=None):
    nest_config = find("nest-config")
    if not nest_config:
        warnings.warn("Cannot find nest-config, please check your PATH. Unable to build extensions.")
        return

    logger.debug("nest-config found at", nest_config)

    build_dirs = []
    if build_dir is not None:
        build_dirs.append(build_dir)
    # if a specific build directory is not provided,
    # first try to build within the pyNN source dir
    build_dirs.append(os.path.join(os.path.dirname(__file__), "_build"))
    # if that directory is not writable, build in the current working directory
    build_dirs.append(os.path.join(os.getcwd(), "_build", "nest_extensions"))

    for nest_build_dir in build_dirs:
        try:
            os.makedirs(nest_build_dir, exist_ok=True)
        except OSError:
            continue
        if os.access(nest_build_dir, os.W_OK):
            break

    if not os.access(nest_build_dir, os.W_OK):
        warnings.warn("Cannot create build directory for nest extensions")
        return

    source_dir = os.path.join(os.path.dirname(__file__), "extensions")
    result, stdout = run_command(f"cmake -Dwith-nest={nest_config} {source_dir}",
                                 nest_build_dir)
    if result != 0:
        err_msg = "\n  ".join(stdout)
        warnings.warn(f"Problem running cmake. Output was:\n  {err_msg}")
    else:
        result, stdout = run_command("make", nest_build_dir)
        if result != 0:
            err_msg = "\n  ".join(stdout)
            warnings.warn(f"Unable to compile NEST extensions. Output was:\n  {err_msg}")
        else:
            result, stdout = run_command("make install", nest_build_dir)
            if result != 0:
                err_msg = "\n  ".join(stdout)
                warnings.warn(f"Unable to install NEST extensions. Output was:\n  {err_msg}")
            else:
                logger.info("Successfully compiled NEST extensions.")


# --- For implementation of get_time_step() and similar functions --------------


def nest_property(name, dtype):
    """Return a property that accesses a NEST kernel parameter"""

    def _get(self):
        return nest.GetKernelStatus(name)

    def _set(self, val):
        try:
            nest.SetKernelStatus({name: dtype(val)})
        except nest.NESTError as e:
            reraise(e, "%s = %s (%s)" % (name, val, type(val)))
    return property(fget=_get, fset=_set)


def apply_time_offset(parameters, offset):
    parameters_copy = {}
    for name, value in parameters.items():
        if name in NEST_VARIABLES_TIME_DIMENSION:
            parameters_copy[name] = value + offset
        elif name in NEST_ARRAY_VARIABLES_TIME_DIMENSION:
            parameters_copy[name] = [v + offset for v in value]
        else:
            parameters_copy[name] = value
    return parameters_copy


class _State(common.control.BaseState):
    """Represent the simulator state."""

    def __init__(self):
        super().__init__()
        try:
            nest.Install('pynn_extensions')
            self.extensions_loaded = True
        except nest.NESTError:
            build_extensions()
            try:
                nest.Install('pynn_extensions')
                self.extensions_loaded = True
            except nest.NESTError:
                self.extensions_loaded = False
        self.initialized = False
        self.optimize = False
        self.spike_precision = "off_grid"
        self.verbosity = "error"
        # the following line avoids blocking if only some nodes call num_processes
        # do the same for rank?
        self._cache_num_processes = nest.GetKernelStatus()['num_processes']
        # allow NEST to erase previously written files (defaut with all the other simulators)
        nest.SetKernelStatus({'overwrite_files': True})
        self.tempdirs = []
        self.recording_devices = []
        self.populations = []  # needed for reset
        self.current_sources = []
        self._time_offset = 0.0
        self.t_flush = -1
        self.stale_connection_cache = False

    @property
    def t(self):
        # note that we always simulate one min delay past the requested time
        # we round to try to reduce floating-point problems
        # longer-term, we should probably work with integers (in units of time step)
        return max(np.around(self.t_kernel - self.min_delay - self._time_offset, decimals=12), 0.0)

    t_kernel = nest_property("biological_time", float)

    dt = nest_property('resolution', float)

    threads = nest_property('local_num_threads', int)

    rng_seed = nest_property('rng_seed', int)
    grng_seed = nest_property('rng_seed', int)

    @property
    def min_delay(self):
        return nest.GetKernelStatus('min_delay')

    def set_delays(self, min_delay, max_delay):
        # this assumes we never set max_delay without also setting min_delay
        if min_delay != 'auto':
            min_delay = float(min_delay)
            if max_delay == 'auto':
                max_delay = 10.0
            else:
                max_delay = float(max_delay)
            nest.SetKernelStatus({'min_delay': min_delay,
                                  'max_delay': max_delay})

    @property
    def max_delay(self):
        return nest.GetKernelStatus('max_delay')

    @property
    def num_processes(self):
        return self._cache_num_processes

    @property
    def mpi_rank(self):
        return nest.Rank()

    def _get_spike_precision(self):
        return self._spike_precision

    def _set_spike_precision(self, precision):
        if nest.off_grid_spiking and precision == "on_grid":
            raise ValueError(
                "The option to use off-grid spiking cannot be turned off once enabled")
        if precision == 'off_grid':
            self.default_recording_precision = 15
        elif precision == 'on_grid':
            self.default_recording_precision = 3
        else:
            raise ValueError("spike_precision must be 'on_grid' or 'off_grid'")
        self._spike_precision = precision
    spike_precision = property(fget=_get_spike_precision, fset=_set_spike_precision)

    def _set_verbosity(self, verbosity):
        nest.set_verbosity('M_{}'.format(verbosity.upper()))
    verbosity = property(fset=_set_verbosity)

    def set_status(self, nodes, params, val=None):
        """
        Wrapper around nest.SetStatus() to handle time offset
        """
        if self._time_offset == 0.0:
            nest.SetStatus(nodes, params, val=val)
        else:
            if val is None:
                parameters = params
            else:
                parameters = {params: val}

            if isinstance(parameters, list):
                params_copy = []
                for item in parameters:
                    params_copy.append(apply_time_offset(item, self._time_offset))
            else:
                params_copy = apply_time_offset(parameters, self._time_offset)
            nest.SetStatus(nodes, params_copy)

    def run(self, simtime):
        """Advance the simulation for a certain time."""
        for population in self.populations:
            if population._deferred_parrot_connections:
                population._connect_parrot_neurons()
        for device in self.recording_devices:
            if not device._connected:
                device.connect_to_cells()
                device._local_files_merged = False
        if not self.running and simtime > 0:
            # we simulate past the real time by one min_delay,
            # otherwise NEST doesn't give us all the recorded data
            simtime += self.min_delay
            self.running = True
        if simtime > 0:
            nest.Simulate(simtime)

    def run_until(self, tstop):
        self.run(tstop - self.t)

    def reset(self):
        if self.t > 0:
            if self.t_flush < 0:
                raise ValueError(
                    "Full reset functionality is not currently available with NEST. "
                    "If you nevertheless want to use this functionality, pass the `t_flush`"
                    "argument to `setup()` with a suitably large value (>> 100 ms)"
                    "then check carefully that the previous run is not influencing the "
                    "following one."
                )
            else:
                warnings.warn(
                    "Full reset functionality is not available with NEST. "
                    "Please check carefully that the previous run is not influencing the "
                    "following one and, if so, increase the `t_flush` argument to `setup()`"
                )
            self.run(self.t_flush)  # get spikes and recorded data out of the system
            for recorder in self.recorders:
                recorder._clear_simulator()

        self._time_offset = self.t_kernel

        for p in self.populations:
            if hasattr(p.celltype, "uses_parrot") and p.celltype.uses_parrot:
                # 'uses_parrot' is a marker for spike sources,
                # which may have parameters that need to be updated
                # to account for time offset
                # TODO: need to ensure that get/set parameters also works correctly
                p._set_parameters(p.celltype.native_parameters)
            for variable, initial_value in p.initial_values.items():
                p._set_initial_value_array(variable, initial_value)
                p._reset()
        for cs in self.current_sources:
            cs._reset()

        self.running = False
        self.segment_counter += 1

    def clear(self):
        self.populations = []
        self.current_sources = []
        self.recording_devices = []
        self.recorders = set()
        # clear the sli stack, if this is not done --> memory leak cause the stack increases
        nest.ll_api.sr('clear')
        # reset the simulation kernel
        nest.ResetKernel()
        # but this reverts some of the PyNN settings, so we have to repeat them (see NEST #716)
        self.spike_precision = self._spike_precision
        # set tempdir
        tempdir = tempfile.mkdtemp()
        self.tempdirs.append(tempdir)  # append tempdir to tempdirs list
        nest.SetKernelStatus({'data_path': tempdir, })
        self.segment_counter = -1
        self.reset()


# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)

    @property
    def local(self):
        return self.node_collection.local


# --- For implementation of connect() and Connector classes --------------------

class Connection(common.Connection):
    """
    Provide an interface that allows access to the connection's weight, delay
    and other attributes.
    """

    def __init__(self, parent, index):
        """
        Create a new connection interface.

        `parent` -- a Projection instance.
        `index` -- the index of this connection in the parent.
        """
        self.parent = parent
        self.index = index

    def id(self):
        """Return a tuple of arguments for `nest.GetConnection()`.
        """
        return self.parent.nest_connections[self.index]

    @property
    def source(self):
        """The ID of the pre-synaptic neuron."""
        src = ID(nest.GetStatus(self.id(), 'source')[0])
        src.parent = self.parent.pre
        return src
    presynaptic_cell = source

    @property
    def target(self):
        """The ID of the post-synaptic neuron."""
        tgt = ID(nest.GetStatus(self.id(), 'target')[0])
        tgt.parent = self.parent.post
        return tgt
    postsynaptic_cell = target

    def _set_weight(self, w):
        nest.SetStatus(self.id(), 'weight', w * 1000.0)

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        w_nA = nest.GetStatus(self.id(), 'weight')[0]
        if self.parent.synapse_type == 'inhibitory' and common.is_conductance(self.target):
            # NEST uses negative values for inhibitory weights,
            # even if these are conductances
            w_nA *= -1
        return 0.001 * w_nA

    def _set_delay(self, d):
        nest.SetStatus(self.id(), 'delay', d)

    def _get_delay(self):
        """Synaptic delay in ms."""
        return nest.GetStatus(self.id(), 'delay')[0]

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)


def generate_synapse_property(name):
    def _get(self):
        return nest.GetStatus(self.id(), name)[0]

    def _set(self, val):
        nest.SetStatus(self.id(), name, val)
    return property(_get, _set)


setattr(Connection, 'U', generate_synapse_property('U'))
setattr(Connection, 'tau_rec', generate_synapse_property('tau_rec'))
setattr(Connection, 'tau_facil', generate_synapse_property('tau_fac'))
setattr(Connection, 'u0', generate_synapse_property('u0'))
setattr(Connection, '_tau_psc', generate_synapse_property('tau_psc'))


# --- Initialization, and module attributes ------------------------------------

state = _State()  # a Singleton, so only a single instance ever exists
del _State
