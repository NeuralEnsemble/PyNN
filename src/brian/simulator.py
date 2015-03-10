# encoding: utf-8
"""
Implementation of the "low-level" functionality used by the common
implementation of the API, for the Brian simulator.

Classes and attributes usable by the common implementation:

Classes:
    ID
    Connection

Attributes:
    state -- an instance of the _State class.

All other functions and classes are private, and should not be used by other
modules.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import brian
import numpy
from pyNN import common

name = "Brian"
logger = logging.getLogger("PyNN")

ms = brian.ms


class ID(int, common.IDMixin):
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


class State(common.control.BaseState):
    
    def __init__(self):
        common.control.BaseState.__init__(self)
        self.mpi_rank = 0
        self.num_processes = 1
        self.network = None
        self._min_delay = 'auto'
        self.clear()
    
    def run(self, simtime):
        self.running = True
        self.network.run(simtime * ms)
        
    def run_until(self, tstop):
        self.run(tstop - self.t)
        
    def clear(self):
        self.recorders = set([])
        self.id_counter = 0
        self.segment_counter = -1
        if self.network:
            for item in self.network.groups + self.network._all_operations:
                del item
        self.network = brian.Network()
        self.network.clock = brian.Clock()
        self.reset()
        
    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.network.reinit()
        self.running = False
        self.t_start = 0
        self.segment_counter += 1
        for group in self.network.groups:
            if hasattr(group, "initialize"):
                logger.debug("Re-initalizing %s" % group)
                group.initialize()
        
    def _get_dt(self):
        if self.network.clock is None:
            raise Exception("Simulation timestep not yet set. Need to call setup()")
        return float(self.network.clock.dt/ms)

    def _set_dt(self, timestep):
        logger.debug("Setting timestep to %s", timestep)
        #if self.network.clock is None or timestep != self._get_dt():
        #    self.network.clock = brian.Clock(dt=timestep*ms)
        self.network.clock.dt = timestep * ms
    dt = property(fget=_get_dt, fset=_set_dt)

    @property
    def t(self):
        return float(self.network.clock.t/ms)

    def _get_min_delay(self):
        if self._min_delay == 'auto':
            min_delay = numpy.inf
            for item in self.network.groups:
                if isinstance(item, brian.Synapses):
                    min_delay = min(min_delay, item.delay.to_matrix().min())
            if numpy.isinf(min_delay):
                self._min_delay = self.dt
            else:
                self._min_delay = min_delay * self.dt  # Synapses.delay is an integer, the number of time steps
        return self._min_delay
    def _set_min_delay(self, delay):
        self._min_delay = delay
    min_delay = property(fget=_get_min_delay, fset=_set_min_delay)


state = State()
