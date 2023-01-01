# ~mc/pyNN/arbor/simulator.py
# encoding: utf8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the Arbor simulator.
Classes and attributes useable by the common implementation:
Classes:
    ID
    Connection
Attributes:
    state -- a singleton instance of the _State class.
All other functions and classes are private, and should not be used by other
modules.
:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    xrange
except NameError:
    xrange = range
from pyNN import __path__ as pyNN_path
from pyNN import common
from pyNN.morphology import MorphologyFilter
import logging
import numpy
import os.path
#from neuron import h, nrn_dll_loaded
from operator import itemgetter

logger = logging.getLogger("PyNN")
name = "Arbor"  # for use in annotating output data

# Instead of starting the projection var-GID range from 0, the first _MIN_PROJECTION_VARGID are 
# reserved for other potential uses
_MIN_PROJECTION_VARGID = 1000000 

# --- For implementation of get_time_step() and similar functions --------------

class _State(common.control.BaseState):
    """Represent the simulator state."""

    def __init__(self):
        """Initialize the simulator."""
        super(_State, self).__init__()
        self.clear()
        self.dt = 0.025 # default in parent Arbor
        self.default_maxstep = 10.0
        self.native_rng_baseseed = 0

#    def __get_dt(self):
#        return self.dt

#    def __set_dt(self, dt):
#        self.dt = dt
        
#    dt = property(fget=__get_dt, fset=__set_dt)
    
    def clear(self):
        #self.parallel_context.gid_clear()
        #self.gid_sources = []
        #self.recorders = set([])
        #self.current_sources = []
        #self.gid_counter = 0
        #self.vargid_offsets = dict()  # Contains the start of the available "variable"-GID range for each projection (as opposed to "cell"-GIDs)
        self.reset()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.tfinal = 0
        #self.tstop = 0

#    def _pre_run(self):
#        if not self.running:
#            self.running = True
#            local_minimum_delay = self.parallel_context.set_maxstep(self.default_maxstep)
#            if state.vargid_offsets:
#                logger.info("Setting up transfer on MPI process {}".format(state.mpi_rank))
#                state.parallel_context.setup_transfer()
#            h.finitialize()
#            self.tstop = 0
#            logger.debug("default_maxstep on host #%d = %g" % (self.mpi_rank, self.default_maxstep))
#            logger.debug("local_minimum_delay on host #%d = %g" % (self.mpi_rank, local_minimum_delay))
#            if self.min_delay == 'auto':
#                self.min_delay = local_minimum_delay
#            else:
#                if self.num_processes > 1:
#                    assert local_minimum_delay >= self.min_delay, \
#                       "There are connections with delays (%g) shorter than the minimum delay (%g)" % (local_minimum_delay, self.min_delay)

#    def _update_current_sources(self, tstop):
#        for source in self.current_sources:
#            for iclamp in source._devices:
#                source._update_iclamp(iclamp, tstop)

#    def run(self, simtime):
#        """Advance the simulation for a certain time."""
#        self.run_until(self.tstop + simtime)

#    def run_until(self, tstop):
#        self._update_current_sources(tstop)
#        self._pre_run()
#        self.tstop = tstop
#        #logger.info("Running the simulation until %g ms" % tstop)
#        if self.tstop > self.t:
#            self.parallel_context.psolve(self.tstop)

#    def finalize(self, quit=False):
#        """Finish using NEURON."""
#        self.parallel_context.runworker()
#        self.parallel_context.done()
#        if quit:
#            logger.info("Finishing up with NEURON.")
#            h.quit()

#    def get_vargids(self, projection, pre_idx, post_idx):
#        """
#        Get new "variable"-GIDs (as opposed to the "cell"-GIDs) for a given pre->post connection 
#        pair for a given projection. 
#        `projection`  -- projection
#        `pre_idx`     -- index of the presynaptic cell
#        `post_idx`     -- index of the postsynaptic cell
#        """
#        try:
#            offset = self.vargid_offsets[projection]
#        except KeyError:
#            # Get the projection with the current maximum vargid offset
#            if len(self.vargid_offsets):
#                newest_proj, offset = max(self.vargid_offsets.items(), key=itemgetter(1))
#                # Allocate it a large enough range for a mutual all-to-all connection (assumes that 
#                # there are no duplicate pre_idx->post_idx connections for the same projection. If
#                # that is really desirable a new projection will need to be used)
#                offset += 2 * len(newest_proj.pre) * len(newest_proj.post)
#            else:
#                offset = _MIN_PROJECTION_VARGID 
#            self.vargid_offsets[projection] = offset
#        pre_post_vargid = offset + 2 * (pre_idx + post_idx * len(projection.pre))
#        post_pre_vargid = pre_post_vargid + 1
#        return (pre_post_vargid, post_pre_vargid)

# --- Initialization, and module attributes ------------------------------------

state = _State()  # a Singleton, so only a single instance ever exists
del _State