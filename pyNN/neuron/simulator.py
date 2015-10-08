# encoding: utf8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the NEURON simulator.

Classes and attributes useable by the common implementation:

Classes:
    ID
    Connection

Attributes:
    state -- a singleton instance of the _State class.

All other functions and classes are private, and should not be used by other
modules.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

try:
    xrange
except NameError:
    xrange = range
from pyNN import __path__ as pyNN_path
from pyNN import common
import logging
import numpy
import os.path
from neuron import h, nrn_dll_loaded
from operator import itemgetter

logger = logging.getLogger("PyNN")
name = "NEURON"  # for use in annotating output data

# Instead of starting the projection var-GID range from 0, the first _MIN_PROJECTION_VARGID are 
# reserved for other potential uses
_MIN_PROJECTION_VARGID = 1000000 

# --- Internal NEURON functionality --------------------------------------------

def load_mechanisms(path):
    """
    Search for and load NMODL mechanisms from the path given.

    This a stricter version of NEURON's own load_mechanisms function, which will
    raise an IOError if no mechanisms are found at the given path. This function
    will not load a mechanism path twice.

    The path should specify the directory in which nrnivmodl was run, and in
    which the directory 'i686' (or 'x86_64' or 'powerpc' depending on your
    platform) was created.
    """
    import platform

    global nrn_dll_loaded
    if path in nrn_dll_loaded:
        logger.warning("Mechanisms already loaded from path: %s" % path)
        return
    # in case NEURON is assuming a different architecture to Python,
    # we try multiple possibilities
    arch_list = [platform.machine(), 'i686', 'x86_64', 'powerpc', 'umac']
    for arch in arch_list:
        lib_path = os.path.join(path, arch, '.libs', 'libnrnmech.so')
        if os.path.exists(lib_path):
            h.nrn_load_dll(lib_path)
            nrn_dll_loaded.append(path)
            return
    raise IOError("NEURON mechanisms not found in %s. You may need to run 'nrnivmodl' in this directory." % path)


def is_point_process(obj):
    """Determine whether a particular object is a NEURON point process."""
    return hasattr(obj, 'loc')


def nativeRNG_pick(n, rng, distribution='uniform', parameters=[0,1]):
    """
    Pick random numbers from a Hoc Random object.

    Return a Numpy array.
    """
    native_rng = h.Random(0 or rng.seed)
    rarr = [getattr(native_rng, distribution)(*parameters)]
    rarr.extend([native_rng.repick() for j in xrange(n-1)])
    return numpy.array(rarr)


def h_property(name):
    """Return a property that accesses a global variable in Hoc."""
    def _get(self):
        return getattr(h, name)
    def _set(self, val):
        setattr(h, name, val)
    return property(fget=_get, fset=_set)


class _Initializer(object):
    """
    Manage initialization of NEURON cells. Rather than create an
    `FInializeHandler` instance for each cell that needs to initialize itself,
    we create a single instance, and use an instance of this class to maintain
    a list of cells that need to be initialized.

    Public methods:
        register()
    """

    def __init__(self):
        """
        Create an `FinitializeHandler` object in Hoc, which will call the
        `_initialize()` method when NEURON is initialized.
        """
        h('objref initializer')
        h.initializer = self
        self.fih = h.FInitializeHandler(1, "initializer._initialize()")
        self.clear()

    def register(self, *items):
        """
        Add items to the list of cells/populations to be initialized. Cell
        objects must have a `memb_init()` method.
        """
        for item in items:
            if isinstance(item, (common.BasePopulation, common.Assembly)):
                if item.celltype.injectable: # don't do memb_init() on spike sources
                    self.population_list.append(item)
            else:
                if hasattr(item._cell, "memb_init"):
                    self.cell_list.append(item)

    def _initialize(self):
        """Call `memb_init()` for all registered cell objects."""
        logger.info("Initializing membrane potential of %d cells and %d Populations." % \
                     (len(self.cell_list), len(self.population_list)))
        for cell in self.cell_list:
            cell._cell.memb_init()
        for population in self.population_list:
            for cell in population:
                cell._cell.memb_init()

    def clear(self):
        self.cell_list = []
        self.population_list = []


# --- For implementation of get_time_step() and similar functions --------------

class _State(common.control.BaseState):
    """Represent the simulator state."""

    def __init__(self):
        """Initialize the simulator."""
        super(_State, self).__init__()
        h('min_delay = -1')
        h('tstop = 0')
        h('steps_per_ms = 1/dt')
        self.parallel_context = h.ParallelContext()
        self.parallel_context.spike_compress(1, 0)
        self.num_processes = int(self.parallel_context.nhost())
        self.mpi_rank = int(self.parallel_context.id())
        self.cvode = h.CVode()
        h('objref plastic_connections')
        self.clear()
        self.default_maxstep = 10.0
        self.native_rng_baseseed  = 0

    t = h_property('t')
    def __get_dt(self):
        return h.dt
    def __set_dt(self, dt):
        h.steps_per_ms = 1.0/dt
        h.dt = dt
    dt = property(fget=__get_dt, fset=__set_dt)
    tstop = h_property('tstop')         # these are stored in hoc so that we

    def __set_min_delay(self, val):     # can interact with the GUI
        if val != 'auto':
            h.min_delay = val
    def __get_min_delay(self):
        if h.min_delay < 0:
            return 'auto'
        else:
            return h.min_delay
    min_delay = property(fset=__set_min_delay, fget=__get_min_delay)

    def register_gid(self, gid, source, section=None):
        """Register a global ID with the global `ParallelContext` instance."""
        ###print("registering gid %s to %s (section=%s)" % (gid, source, section))
        self.parallel_context.set_gid2node(gid, self.mpi_rank) # assign the gid to this node
        if is_point_process(source):
            nc = h.NetCon(source, None)                          # } associate the cell spike source
        else:
            nc = h.NetCon(source, None, sec=section)
        self.parallel_context.cell(gid, nc)                     # } with the gid (using a temporary NetCon)
        self.gid_sources.append(source) # gid_clear (in _State.reset()) will cause a
                                        # segmentation fault if any of the sources
                                        # registered using pc.cell() no longer exist, so
                                        # we keep a reference to all sources in the
                                        # global gid_sources list. It would be nicer to
                                        # be able to unregister a gid and have a __del__
                                        # method in ID, but this will do for now.

    def clear(self):
        self.parallel_context.gid_clear()
        self.gid_sources = []
        self.recorders = set([])
        self.gid_counter = 0
        self.vargid_offsets = dict() # Contains the start of the available "variable"-GID range for each projection (as opposed to "cell"-GIDs)
        h.plastic_connections = []
        self.segment_counter = -1
        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.tstop = 0
        self.t_start = 0
        self.segment_counter += 1
        h.finitialize()

    def _pre_run(self):
        if not self.running:
            self.running = True
            local_minimum_delay = self.parallel_context.set_maxstep(self.default_maxstep)
            if state.vargid_offsets:
                logger.info("Setting up transfer on MPI process {}".format(state.mpi_rank))
                state.parallel_context.setup_transfer()
            h.finitialize()
            self.tstop = 0
            logger.debug("default_maxstep on host #%d = %g" % (self.mpi_rank, self.default_maxstep ))
            logger.debug("local_minimum_delay on host #%d = %g" % (self.mpi_rank, local_minimum_delay))
            if self.min_delay == 'auto':
                self.min_delay = local_minimum_delay
            else:
                if self.num_processes > 1:
                    assert local_minimum_delay >= self.min_delay, \
                       "There are connections with delays (%g) shorter than the minimum delay (%g)" % (local_minimum_delay, self.min_delay)

    def run(self, simtime):
        """Advance the simulation for a certain time."""
        self.run_until(self.tstop + simtime)

    def run_until(self, tstop):
        self._pre_run()
        self.tstop = tstop
        #logger.info("Running the simulation until %g ms" % tstop)
        if self.tstop > self.t:
            self.parallel_context.psolve(self.tstop)

    def finalize(self, quit=False):
        """Finish using NEURON."""
        self.parallel_context.runworker()
        self.parallel_context.done()
        if quit:
            logger.info("Finishing up with NEURON.")
            h.quit()

    def get_vargids(self, projection, pre_idx, post_idx):
        """
        Get new "variable"-GIDs (as opposed to the "cell"-GIDs) for a given pre->post connection 
        pair for a given projection. 

        `projection`  -- projection
        `pre_idx`     -- index of the presynaptic cell
        `post_idx`     -- index of the postsynaptic cell
        """
        try:
            offset = self.vargid_offsets[projection]
        except KeyError:
            # Get the projection with the current maximum vargid offset
            if len(self.vargid_offsets):
                newest_proj, offset = max(self.vargid_offsets.items(), key=itemgetter(1))
                # Allocate it a large enough range for a mutual all-to-all connection (assumes that 
                # there are no duplicate pre_idx->post_idx connections for the same projection. If
                # that is really desirable a new projection will need to be used)
                offset += 2 * len(newest_proj.pre) * len(newest_proj.post)
            else:
                offset = _MIN_PROJECTION_VARGID 
            self.vargid_offsets[projection] = offset
        pre_post_vargid = offset + 2 * (pre_idx + post_idx * len(projection.pre))
        post_pre_vargid = pre_post_vargid + 1
        return (pre_post_vargid, post_pre_vargid)

# --- For implementation of access to individual neurons' parameters -----------

class ID(int, common.IDMixin):
    __doc__ = common.IDMixin.__doc__

    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)

    def _build_cell(self, cell_model, cell_parameters):
        """
        Create a cell in NEURON, and register its global ID.

        `cell_model` -- one of the cell classes defined in the
                        `neuron.cells` module (more generally, any class that
                        implements a certain interface, but I haven't
                        explicitly described that yet).
        `cell_parameters` -- a ParameterSpace containing the parameters used to
                             initialise the cell model.
        """
        gid = int(self)
        self._cell = cell_model(**cell_parameters)          # create the cell object
        state.register_gid(gid, self._cell.source, section=self._cell.source_section)
        if hasattr(self._cell, "get_threshold"):            # this is not adequate, since the threshold may be changed after cell creation
            state.parallel_context.threshold(int(self), self._cell.get_threshold()) # the problem is that self._cell does not know its own gid

    def get_initial_value(self, variable):
        """Get the initial value of a state variable of the cell."""
        return getattr(self._cell, "%s_init" % variable)

    def set_initial_value(self, variable, value):
        """Set the initial value of a state variable of the cell."""
        index = self.parent.id_to_local_index(self)
        self.parent.initial_values[variable][index] = value
        setattr(self._cell, "%s_init" % variable, value)


class Connection(common.Connection):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, projection, pre, post, **parameters):
        """
        Create a new connection.
        """
        #logger.debug("Creating connection from %d to %d, weight %g" % (pre, post, parameters['weight']))
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.presynaptic_cell = projection.pre[pre]
        self.postsynaptic_cell = projection.post[post]
        if "." in projection.receptor_type:
            section, target = projection.receptor_type.split(".")
            target_object = getattr(getattr(self.postsynaptic_cell._cell, section), target)
        else:
            target_object = getattr(self.postsynaptic_cell._cell, projection.receptor_type)
        self.nc = state.parallel_context.gid_connect(int(self.presynaptic_cell), target_object)
        self.nc.weight[0] = parameters.pop('weight')
        # if we have a mechanism (e.g. from 9ML) that includes multiple
        # synaptic channels, need to set nc.weight[1] here
        if self.nc.wcnt() > 1 and hasattr(self.postsynaptic_cell._cell, "type"):
            self.nc.weight[1] = self.postsynaptic_cell._cell.type.receptor_types.index(projection.receptor_type)
        self.nc.delay  = parameters.pop('delay')
        if projection.synapse_type.model is not None:
            self._setup_plasticity(projection.synapse_type, parameters)
        # nc.threshold is supposed to be set by ParallelContext.threshold, called in _build_cell(), above, but this hasn't been tested

    def _setup_plasticity(self, synapse_type, parameters):
        """
        Set this connection to use spike-timing-dependent plasticity.

        `mechanism`  -- the name of an NMODL mechanism that modifies synaptic
                        weights based on the times of pre- and post-synaptic spikes.
        `parameters` -- a dictionary containing the parameters of the weight-
                        adjuster mechanism.
        """
        mechanism = synapse_type.model
        self.weight_adjuster = getattr(h, mechanism)(0.5)
        if synapse_type.postsynaptic_variable == 'spikes':
            parameters['allow_update_on_post'] = int(False) # for compatibility with NEST
            self.ddf = parameters.pop('dendritic_delay_fraction')
            # If ddf=1, the synaptic delay
            # `d` is considered to occur entirely in the post-synaptic
            # dendrite, i.e., the weight adjuster receives the pre-
            # synaptic spike at the time of emission, and the post-
            # synaptic spike a time `d` after emission. If ddf=0, the
            # synaptic delay is considered to occur entirely in the
            # pre-synaptic axon.
        elif synapse_type.postsynaptic_variable is None:
            self.ddf = 0
        else:
            raise NotImplementedError("Only post-synaptic-spike-dependent mechanisms available for now.")
        self.pre2wa = state.parallel_context.gid_connect(int(self.presynaptic_cell), self.weight_adjuster)
        self.pre2wa.threshold = self.nc.threshold
        self.pre2wa.delay = self.nc.delay * (1 - self.ddf)
        if self.pre2wa.delay > 1e-9:
            self.pre2wa.delay -= 1e-9   # we subtract a small value so that the synaptic weight gets updated before it is used.
        if synapse_type.postsynaptic_variable == 'spikes':
            # directly create NetCon as wa is on the same machine as the post-synaptic cell
            self.post2wa = h.NetCon(self.postsynaptic_cell._cell.source, self.weight_adjuster,
                                    sec=self.postsynaptic_cell._cell.source_section)
            self.post2wa.threshold = 1
            self.post2wa.delay = self.nc.delay * self.ddf
            self.post2wa.weight[0] = -1
            self.pre2wa.weight[0] = 1
        else:
            self.pre2wa.weight[0] = self.nc.weight[0]
        parameters.pop('x', None)   # for the Tsodyks-Markram model
        parameters.pop('y', None)   # would be better to actually use these initial values
        for name, value in parameters.items():
            setattr(self.weight_adjuster, name, value)
        if mechanism == 'TsodyksMarkramWA':  # or could assume that any weight_adjuster parameter called "tau_syn" should be set like this
            self.weight_adjuster.tau_syn = self.nc.syn().tau
        # setpointer
        i = len(h.plastic_connections)
        h.plastic_connections.append(self)
        h('setpointer plastic_connections._[%d].weight_adjuster.wsyn, plastic_connections._[%d].nc.weight' % (i, i))

    def _set_weight(self, w):
        self.nc.weight[0] = w

    def _get_weight(self):
        """Synaptic weight in nA or µS."""
        return self.nc.weight[0]

    def _set_delay(self, d):
        self.nc.delay = d
        if hasattr(self, 'pre2wa'):
            self.pre2wa.delay = float(d)*(1-self.ddf)
        if hasattr(self, 'post2wa'):
            self.post2wa.delay = float(d)*self.ddf

    def _get_delay(self):
        """Connection delay in ms."""
        return self.nc.delay

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)

    def as_tuple(self, *attribute_names):
        # need to do translation of names, or perhaps that should be handled in common?
        return tuple(getattr(self, name) for name in attribute_names)


class GapJunction(object):
    """
    Store an individual gap junction connection and information about it. Provide an
    interface that allows access to the connection's conductance attributes
    """

    def __init__(self, projection, pre, post, **parameters):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        segment_name = projection.receptor_type
        # Strip 'gap' string from receptor_type (not sure about this, it is currently appended to
        # the available synapse types in the NCML model segments but is not really necessary and 
        # it feels a bit hacky but it makes the list of receptor types more comprehensible)
        if segment_name.endswith('.gap'): 
            segment_name = segment_name[:-4]
        self.segment = getattr(projection.post[post]._cell, segment_name)
        pre_post_vargid, post_pre_vargid = state.get_vargids(projection, pre, post)
        self._make_connection(self.segment, parameters.pop('weight'), pre_post_vargid,   
                              post_pre_vargid, projection.pre[pre], projection.post[post])
        
    def _make_connection(self, segment, weight, local_to_remote_vargid, remote_to_local_vargid,
                         local_gid, remote_gid):       
        logger.debug("Setting source_var on local cell {} to connect to target_var on remote "
                     "cell {} with vargid {} on process {}"
                    .format(local_gid, remote_gid, local_to_remote_vargid, 
                            state.mpi_rank))
        # Set up the source reference for the local->remote connection 
        state.parallel_context.source_var(segment(0.5)._ref_v, local_to_remote_vargid)              
        # Create the gap_junction and set its weight
        self.gap = h.Gap(0.5, sec=segment)
        self.gap.g = weight
        # Connect the gap junction with the source_var
        logger.debug("Setting target_var on local cell {} to connect to source_var on remote "
                     "cell {} with vargid {} on process {}"
                    .format(local_gid, remote_gid, remote_to_local_vargid, 
                            state.mpi_rank))
        # set up the target reference for the remote->local connection
        state.parallel_context.target_var(self.gap._ref_vgap, remote_to_local_vargid)
        
    def _set_weight(self, w):
        self.gap.g = w

    def _get_weight(self):
        """Gap junction conductance in µS."""
        return self.gap.g
    
    weight = property(_get_weight, _set_weight)
    
    def as_tuple(self, *attribute_names):
        return tuple(getattr(self, name) for name in attribute_names)
    
class GapJunctionPresynaptic(GapJunction):
    """
    The presynaptic component of a gap junction. Gap junctions in NEURON are actually symmetrical
    so it shares its functionality with the GapJunction connection object, with the exception that
    the pre and post synaptic cells are switched
    """
    
    def __init__(self, projection, pre, post, **parameters):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        if projection.source.endswith('.gap'): 
            segment_name = projection.source[:-4]
        else:
            segment_name = projection.source
        self.segment = getattr(projection.pre[pre]._cell, segment_name)
        pre_post_vargid, post_pre_vargid = state.get_vargids(projection, pre, post)
        self._make_connection(self.segment, parameters.pop('weight'), post_pre_vargid, 
                              pre_post_vargid, projection.post[post], projection.pre[pre])
    

def generate_synapse_property(name):
    def _get(self):
        return getattr(self.weight_adjuster, name)
    def _set(self, val):
        setattr(self.weight_adjuster, name, val)
    return property(_get, _set)
setattr(Connection, 'w_max', generate_synapse_property('wmax'))
setattr(Connection, 'w_min', generate_synapse_property('wmin'))
setattr(Connection, 'A_plus', generate_synapse_property('aLTP'))
setattr(Connection, 'A_minus', generate_synapse_property('aLTD'))
setattr(Connection, 'tau_plus', generate_synapse_property('tauLTP'))
setattr(Connection, 'tau_minus', generate_synapse_property('tauLTD'))
setattr(Connection, 'U', generate_synapse_property('U'))
setattr(Connection, 'tau_rec', generate_synapse_property('tau_rec'))
setattr(Connection, 'tau_facil', generate_synapse_property('tau_facil'))
setattr(Connection, 'u0', generate_synapse_property('u0'))
setattr(Connection, 'tau', generate_synapse_property('tau'))
setattr(Connection, 'eta', generate_synapse_property('eta'))
setattr(Connection, 'rho', generate_synapse_property('rho'))

# --- Initialization, and module attributes ------------------------------------

mech_path = os.path.join(pyNN_path[0], 'neuron', 'nmodl')
load_mechanisms(mech_path) # maintains a list of mechanisms that have already been imported
state = _State()  # a Singleton, so only a single instance ever exists
del _State
initializer = _Initializer()
del _Initializer
